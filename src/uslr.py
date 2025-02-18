import os
import pdb
import subprocess
from os.path import isfile, join, dirname, basename, exists, isdir
from os import makedirs, listdir
import copy
import warnings
import itertools

from joblib import delayed, Parallel

import numpy as np
import pandas as pd
import nibabel as nib
from skimage.morphology import ball, binary_dilation
import torch
from torch import nn
import surfa as sf
from scipy.optimize import linprog

from src.processing import Processing, GenerativeLabelFusionProcessing
from src.models import InstanceRigidModelLOG, ST2Nonlinear
from src.callbacks import *
from utils.preprocessing_utils import *
from utils.fn_utils import one_hot_encoding, rescale_voxel_size, compute_centroids_ras, gaussian_antialiasing
from utils.label_utils import SYNTHSEG_LUT, CSF_LABELS, SYNTHSEG_APARC_LUT
from utils.def_utils import vol_resample_fast, network_space, create_empty_template, compute_jacobian, getM
from utils.io_utils import create_dir, save_nii
from utils.synthmorph_utils import synthmorph_register, integrate_svf, compose_transforms

from setup import *


class USLRProcessing(Processing):
    def _build_processor(self):
        super()._build_processor()
        self.aff_graph_entities = {'desc': 'raw2temp', 'suffix': 'aff', 'extension': '.npy'}
        self.im_graph_lin_entities = {'space': 'uslrlin', 'acquisition': '1', 'extension': 'nii.gz', 'suffix': 'T1w'}
        self.mask_graph_lin_entities = {'space': 'uslrlin', 'acquisition': '1', 'extension': 'nii.gz',
                                        'suffix': 'T1wmask'}
        self.template_lin_entities = {'space': 'uslrlin', 'acquisition': '1', 'extension': '.nii.gz', 'suffix': 'T1w'}

        self.net_shape = (192, 192, 192)
        self.svf_shape = (96, 96, 96)

        self.net_v2r_entities = {'suffix': 'v2r', 'extension': '.npy', 'space': 'uslrlin', 'desc': 'template'}
        self.svf_v2r_entities = {'suffix': 'v2r', 'extension': '.npy', 'space': 'uslrlin', 'desc': 'svf'}

        self.svf_graph_entities = {'suffix': 'svf', 'extension': 'nii.gz', 'space': 'uslrnonlin', 'scope': 'nonlin'}
        self.template_nonlin_entities = {'space': 'uslrnonlin', 'desc': 'template', 'extension': 'nii.gz'}


class USLRSegment(USLRProcessing):

    def _name(self):
        return 'SynthsegSegmentation'

    def _check_file(self, proxy):
        try:
            if len(proxy.shape) != 3:
                return {'run_flag': False, 'exit_message': 'File excluded due to wrong image dimensions'}


            elif any([s < 20 for s in proxy.shape]):
                return {'run_flag': False, 'exit_message': 'File excluded due to wrong image dimensions'}

            elif any([r > 7 for r in np.sum(np.sqrt(np.abs(proxy.affine * proxy.affine)), axis=0)[:3].tolist()]):
                return {'run_flag': False,
                        'exit_message': 'File excluded due to large resolution in some image dimension.'}

            else:
                return {'run_flag': True, 'exit_message': ''}

        except:
            return {'run_flag': False,
                    'exit_message': 'File excluded due to an error reading the file or computing image shape and resolution.'}

    def _select_image(self, subject, tp):
        # Select a single T1w image per session
        t1w_list = self._get_data(subject=subject, extension='nii.gz', suffix='T1w', session=tp,
                                  acquisition=['orig', None], scope='raw', ignore_check=True)

        if len(t1w_list) == 0:
            return None

        elif len(t1w_list) > 1:
            if any(['acquisition' not in f.entities.keys() for f in t1w_list]):
                t1w_list_r = list(filter(lambda x: 'acquisition' not in x.entities.keys(), t1w_list))

            elif any(['run' in f.entities.keys() for f in t1w_list]):
                t1w_list_r = list(filter(lambda x: x.entities['run'] == '01', t1w_list))

            else:
                t1w_list_r = t1w_list

            t1w_i = t1w_list_r[0]

        else:
            t1w_i = t1w_list[0]

        return t1w_i

    def process_parallel(self, num_cores, **kwargs):
        warnings.warn('Parallel implementation not possible for SynthSeg segmentation. It defers to sequential '
                      'processing')

        return self.process(**kwargs)

    def process(self, prefix='', gpu_flag=False, threads=16, **kwargs):
        self._on_pipeline_init()

        input_files, res_files, output_files, vol_files, discarded_files = [], [], [], [], []
        for subject in self.subject_list:
            output = self.process_subject(subject, **kwargs)
            input_files.extend(output[0])
            res_files.extend(output[1])
            output_files.extend(output[2])
            vol_files.extend(output[3])
            discarded_files.extend(output[4])

        with open(join(TMP_DIR, prefix + '_input_files.txt'), 'w') as f:
            for i_f in input_files:
                f.write(i_f)
                f.write('\n')

        with open(join(TMP_DIR, prefix + '_res_files.txt'), 'w') as f:
            for i_f in res_files:
                f.write(i_f)
                f.write('\n')

        with open(join(TMP_DIR, prefix + '_output_files.txt'), 'w') as f:
            for i_f in output_files:
                f.write(i_f)
                f.write('\n')

        with open(join(TMP_DIR, prefix + '_vol_files.txt'), 'w') as f:
            for i_f in vol_files:
                f.write(i_f)
                f.write('\n')

        if len(output_files) >= 1:
            gpu_cmd = [''] if gpu_flag else ['--cpu']
            subprocess.call(['mri_synthseg',
                             '--i', join(TMP_DIR, prefix + '_input_files.txt'),
                             '--o', join(TMP_DIR, prefix + '_output_files.txt'),
                             '--resample', join(TMP_DIR, prefix + '_res_files.txt'),
                             '--vol', join(TMP_DIR, prefix + '_vol_files.txt'),
                             '--threads', str(threads), '--robust', '--parc'] + gpu_cmd)

        df_new = None
        for subject in listdir(DIR_PIPELINES['preproc']):
            if not isdir(join(DIR_PIPELINES['preproc'], subject)):
                continue

            for sess in listdir(join(DIR_PIPELINES['preproc'], subject)):
                if not exists(join(DIR_PIPELINES['preproc'], subject, sess, 'anat')): continue
                files = list(filter(lambda x: 'T1wdseg.csv' in x or 'T1wdseg.tsv' in x,
                                    listdir(join(DIR_PIPELINES['preproc'], subject, sess, 'anat'))))
                for f in files:
                    df = pd.read_csv(join(DIR_PIPELINES['preproc'], subject, sess, 'anat', f), dtype=str)
                    if len(df.columns) == 1:
                        df = pd.read_csv(join(DIR_PIPELINES['preproc'], subject, sess, 'anat', f), sep='\t', dtype=str)

                    if df_new is None:
                        df_new = pd.DataFrame(columns=['session'] + [c for c in df.columns if 'Unnamed' not in c])

                    if 'session' not in df.columns:
                        for _, row in df.iterrows():
                            try:
                                sid = row['Unnamed: 0'].split('ses-')[-1].split('_')[0]
                                row.drop('Unnamed: 0', inplace=True)
                                row['session'] = sid
                                df_new = pd.concat([df_new, row.to_frame().T])
                            except:
                                pass

                        df_new.set_index('session', drop=False, inplace=True)
                    else:
                        df_new = pd.concat([df_new, df])

            if df_new is not None:
                df_new.to_csv(join(DIR_PIPELINES['preproc'], subject, subject + '_vols.csv'), index=False)
            df_new = None

    def process_subject(self, subject, force_flag=False, check_seg=None, **kwargs):
        if check_seg is None:
            check_seg = '/'

        input_files, res_files, output_files, vol_files, discarded_files = [], [], [], [], []

        timepoints = self.bids_loader.get_session(subject=subject)
        for tp in timepoints:
            # Check if segmentation already exists
            preproc_dirname = join(DIR_PIPELINES['preproc'], 'sub-' + subject, 'ses-' + tp, 'anat')

            t1w_file = self._select_image(subject, tp)
            if t1w_file is None:
                continue

            if not exists(preproc_dirname): os.makedirs(preproc_dirname)
            f = open(join(preproc_dirname, t1w_file.filename.replace('nii.gz', 'txt')), 'w')
            f.write(
                'Since there exists more than one T1w image for this session, we choose this file to run over the '
                'entire USLR pipeline with the corresponding segmentation. Refer to the rawdata to check '
                'correspondence'
            )

            raw_dirname = t1w_file.dirname
            t1w_entities = {k: str(v) for k, v in t1w_file.entities.items() if k in filename_entities}
            t1w_entities['acquisition'] = '1'

            anat_res = basename(self.build_path(t1w_entities))
            anat_seg = anat_res.replace('T1w', 'T1wdseg')
            anat_vols = anat_seg.replace('nii.gz', 'tsv')

            if (exists(join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_seg)) and
                    exists(join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_vols))):
                subprocess.call(['cp', join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_seg), join(preproc_dirname, anat_seg)])
                subprocess.call(['cp', join(check_seg, 'sub-' + subject, 'ses-' + tp, 'anat', anat_vols), join(preproc_dirname, anat_vols)])


            if not exists(join(preproc_dirname, anat_seg)) or force_flag:
                proxy = nib.load(join(raw_dirname, t1w_file.filename))
                run_code = self._check_file(proxy)
                if run_code['run_flag']:
                    input_files += [join(raw_dirname, t1w_file.filename)]
                    res_files += [join(preproc_dirname, anat_res)]
                    output_files += [join(preproc_dirname, anat_seg)]
                    vol_files += [join(preproc_dirname, anat_vols)]
                else:
                    with open(join(preproc_dirname, 'excluded_file.txt'), 'w') as f:
                        f.write(run_code['exit_message'])

        return input_files, res_files, output_files, vol_files, discarded_files


class USLRBiasCorrection(USLRProcessing):

    def _name(self):
        return 'BiasFieldCorrection'

    def _check_resampled_file(self, raw_file, resampled_entities):
        resampled_file = self._get_data(**resampled_entities, ignore_check=True)
        if not resampled_file:
            resampled_filepath = join(DIR_PIPELINES['preproc'], self.build_path(resampled_entities))

            proxyraw = nib.load(raw_file.path)
            pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
            if all([np.abs(p - 1) < 0.01 for p in pixdim]):
                rf = subprocess.call(['ln', '-s', raw_file.path, resampled_filepath], stderr=subprocess.PIPE)
                if rf != 0:
                    subprocess.call(['cp', raw_file.path, resampled_filepath])

            else:
                # some dimension may be wrong
                if any([p < 0.01 for p in pixdim]):
                    return {'exit_code': -1, 'message': 'some dimensions are wrong'}

                v, aff = rescale_voxel_size(np.array(proxyraw.dataobj), proxyraw.affine, [1, 1, 1])
                save_nii(v, aff, resampled_filepath)

        else:
            resampled_filepath = resampled_file[0].path

        return {'exit_code': 0, 'filepath': resampled_filepath}

    def process_subject(self, subject, force_flag=False, remove_wrong=True, **kwargs):
        print('\nSubject: ' + subject)

        timepoints = self._get_timepoints(subject=subject, uslr=True)
        for tp_id in timepoints:
            print('\n* Session: ' + tp_id, end=': ', flush=True)

            preproc_dirname = join(DIR_PIPELINES['preproc'], 'sub-' + subject, 'ses-' + tp_id, 'anat')
            if not exists(preproc_dirname): os.makedirs(preproc_dirname)

            # input segs
            seg_file = self._get_data(**{'session': tp_id, 'subject': subject, **self.seg_entities})
            if seg_file is None:
                continue

            # get entities
            seg_entities = self._get_entities(seg_file)
            seg_entities['extension'] = 'nii.gz'
            raw_entities = {k: str(v) for k, v in seg_entities.items() if k != 'acquisition'}
            raw_entities['suffix'] = 'T1w'
            raw_entities['scope'] = 'raw'
            raw_entities['acquisition'] = [None, 'orig']
            resampled_entities = copy.copy(raw_entities)
            resampled_entities['acquisition'] = '1'
            resampled_entities['scope'] = 'preproc'

            # raw image
            raw_file = self._get_data(**raw_entities)
            if raw_file is None:
                continue

            # build output paths
            output_filepath = join(preproc_dirname, basename(raw_file))
            output_mask_filepath = join(preproc_dirname, seg_file.filename.replace('dseg', 'mask'))

            if exists(output_filepath) and exists(output_mask_filepath) and not force_flag:
                print('image already processed.')
                continue

            # read images
            proxyraw = nib.load(raw_file.path)
            proxyseg = nib.load(seg_file.path)

            # ------------------------ #
            #      Computing masks     #
            # ------------------------ #
            # print('computing masks from dseg files; ', end='', flush=True)
            if not exists(output_mask_filepath):
                seg = np.array(proxyseg.dataobj)
                mask = seg > 0
                for lab in CSF_LABELS:
                    mask[seg == lab] = 0

                save_nii(mask.astype('uint8'), proxyseg.affine, output_mask_filepath)

            resampled_flag = self._check_resampled_file(raw_file, resampled_entities)
            if resampled_flag['exit_code'] == -1:
                print(resampled_flag['message'], end='', flush=True)
                continue

            proxyres = nib.load(resampled_flag['filepath'])

            # ------------------------ #
            # Bias field correction    #
            # ------------------------ #
            # print('correcting for inhomogeneities and normalisation (min/max); ', end='', flush=True)
            if not exists(output_filepath) or force_flag:
                vox2ras0 = proxyres.affine
                mri_acq = np.asarray(proxyres.dataobj)
                mri_acq[np.isnan(mri_acq)] = 0

                pixdimim = np.sqrt(np.sum(proxyres.affine * proxyres.affine, axis=0))[:-1]
                pixdimseg = np.sqrt(np.sum(proxyseg.affine * proxyseg.affine, axis=0))[:-1]
                if any([np.abs(p1-p2)> 0.01 for p1, p2 in zip(pixdimseg, pixdimim)]):
                    proxyseg = vol_resample_fast(proxyres, proxyseg, mode='nearest')

                seg = np.array(proxyseg.dataobj)
                soft_seg = one_hot_encoding(seg, categories=SYNTHSEG_LUT)
                soft_seg = convert_posteriors_to_unified(soft_seg, lut=SYNTHSEG_LUT)
                try:
                    mri_acq_corr, bias_field = bias_field_corr(mri_acq, soft_seg, penalty=1, VERBOSE=False, filter_exceptions=True)
                except:
                    mri_acq_corr = None
                    pdb.set_trace()

                if mri_acq_corr is None:
                    if not remove_wrong:
                        print("[error] bias field cannot be computed -- removing segmentation related files and "
                              "exiting: " + seg_file.path, end='\n')
                        subprocess.call(['rm', '-rf', join(DIR_PIPELINES['preproc'], 'sub-' + subject, 'ses-' + tp_id)])

                    else:
                        print("[error] bias field cannot be computed.", end='', flush=True)

                    continue

                del soft_seg

                mask = seg > 0
                wm_mask = (seg == 2) | (seg == 41)

                del seg

                vox2ras0_orig = proxyraw.affine
                mri_acq_orig = np.asarray(proxyraw.dataobj)
                mri_acq_orig[np.isnan(mri_acq_orig)] = 0
                if len(mri_acq_orig.shape) > 3:
                    mri_acq_orig = mri_acq_orig[..., 0]

                new_vox_size = np.linalg.norm(vox2ras0_orig, 2, 0)[:3]
                vox_size = np.linalg.norm(vox2ras0, 2, 0)[:3]

                if all([v1 == v2 for v1, v2 in zip(vox_size, new_vox_size)]):
                    mask_dilated = binary_dilation(mask, ball(3))
                    m = np.mean(mri_acq_corr[wm_mask])
                    mri_acq_corr = 110 * mri_acq_corr / m
                    mri_acq_corr *= mask_dilated

                    save_nii(np.clip(mri_acq_corr, 0, 255).astype('uint8'), proxyres.affine, output_filepath)

                else:
                    bias_proxy = nib.Nifti1Image(bias_field, proxyres.affine)
                    bias_field_resize = vol_resample_fast(proxyraw, bias_proxy, return_np=True)
                    #
                    mask_proxy = nib.Nifti1Image(mask.astype('float'), proxyres.affine)
                    mask_resize = vol_resample_fast(proxyraw, mask_proxy, return_np=True) > 0.5
                    #
                    wm_mask_proxy = nib.Nifti1Image(wm_mask.astype('float'), proxyres.affine)
                    wm_mask_resize = vol_resample_fast(proxyraw, wm_mask_proxy, return_np=True) > 0.5

                    mri_acq_orig_corr = copy.copy(mri_acq_orig.astype('float32'))
                    mri_acq_orig_corr[mask_resize] = mri_acq_orig_corr[mask_resize] / bias_field_resize[mask_resize]

                    m = np.mean(mri_acq_orig_corr[wm_mask_resize])
                    mri_acq_orig_corr = 110 * mri_acq_orig_corr / m
                    mask_dilated = binary_dilation(mask_resize, ball(3))
                    mri_acq_orig_corr[mask_dilated == 0] = 0

                    save_nii(np.clip(mri_acq_orig_corr, 0, 255).astype('uint8'), proxyraw.affine, output_filepath)

                    del bias_field, bias_field_resize, mri_acq_orig, mri_acq_orig_corr, mask_dilated

            print('done.')


class USLR_PreProcessing(USLRProcessing):

    def __init__(self, bids_loader, subject_list=None):
        super().__init__(bids_loader=bids_loader,
                         subject_list=subject_list)

        self.usrl_segment = USLRSegment(bids_loader, self.subject_list)
        self.usrl_bias_field = USLRBiasCorrection(bids_loader, self.subject_list)


class USLR_Linear(USLRProcessing):

    def _name(self):
        return 'USLR-LinearRegistration'

    def _build_processor(self):
        super()._build_processor()
        self.tmp_dir = join(self.tmp_dir, 'USLR_Lin')
        create_dir(self.tmp_dir)
        self.pipeline_dir = 'lin'

    def _check_running_subject(self, subject, timepoints, force_flag, register_MNI=False):
        # do not run if only 1 timepoint available
        if len(timepoints) == 1:
            if register_MNI:
                return {'exit_code': 5, 'message': '[partly done] It has only 1 timepoint. Linking files and registering to MNI \n'}
            if register_MNI:
                return {'exit_code': 5, 'message': '[done] It has only 1 timepoint. Linking files. \n'}

        # do not run if only 0 timepoint available
        elif len(timepoints) == 0:
            return {'exit_code': 1, 'message': '[done] It has 0 timepoints available. Skipping.\n'}

        # do not run if some timepoint is not properly segmented
        elif any([self._get_data(**{'subject': subject, 'session': t, **self.seg_entities}) is None for t in
                  timepoints]):
            return {'exit_code': -1, 'message': '[error] not all timepoints are correctly segmented. Please check.\n'}

        # do not run if it has already been processed
        elif self._get_data(**{'subject': subject, **self.aff_graph_entities}, curr_len=len(timepoints), verbose=False) is not None and not force_flag:
            filename_sss = self.build_path({'subject': subject, **self.template_lin_entities})

            if not exists(join(DIR_PIPELINES[self.pipeline_dir], filename_sss)):
                return {'exit_code': 2,
                        'message': '[partly done] graph is already computed; template and etiv missing.\n'}
            elif not exists(join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_T1wetiv.npy')):
                return {'exit_code': 3,
                        'message': '[partly done] graph and template are already computed; subject etiv missing.\n'}
            elif self._get_data(**{'subject': subject, 'space': 'MNI', 'suffix': 'T1wdseg'},
                                curr_len=len(timepoints), verbose=False) is None and register_MNI is True:
                return {'exit_code': 4,
                        'message': '[partly done] graph, template and etiv done; MNI registration is missing.\n'}
            else:
                return {'exit_code': 1,
                        'message': '[done] subject already processed. '
                                   'Check the results in [..]/uslr/lin/sub-' + subject + '.\n'}

        # do not run if more segmentations than timepoints are found
        elif self._get_data(**{'subject': subject, **self.seg_entities}, curr_len=len(timepoints)) is None:
            return {'exit_code': -1,
                    'message': '[error] not all timepoints are segmented. Please, run preprocess/synthseg.py first.\n'}

        else:
            return {'exit_code': 0, 'message': ''}

    def _register_timepoints(self, pairwise_centroids, affine_filepath, ok_centr=None):
        # https://www.cse.sc.edu/~songwang/CourseProj/proj2004/ross/ross.pdf

        refCent, floCent = pairwise_centroids

        if ok_centr is not None:
            refCent = refCent[:, ok_centr > 0]
            floCent = floCent[:, ok_centr > 0]

        trans_ref = np.mean(refCent, axis=1, keepdims=True)
        trans_flo = np.mean(floCent, axis=1, keepdims=True)

        refCent_tx = refCent - trans_ref
        floCent_tx = floCent - trans_flo

        cov = refCent_tx @ floCent_tx.T
        u, s, vt = np.linalg.svd(cov)
        I = np.eye(3)
        if np.prod(np.diag(s)) < 0:
            I[-1, -1] = -1

        Q = vt.T @ I @ u.T

        # Full transformation
        Tr = np.eye(4)
        Tr[:3, 3] = -trans_ref.squeeze()

        Tf = np.eye(4)
        Tf[:3, 3] = trans_flo.squeeze()

        R = np.eye(4)
        R[:3, :3] = Q

        aff = Tf @ R @ Tr

        np.save(affine_filepath, aff)

    def _get_centroids(self, subject, timepoints):
        centroid_dict = {}
        ok = {}
        for tp in timepoints:
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': tp})
            centroid_dict[tp], ok[tp] = compute_centroids_ras(seg_file.path, labels_registration)

        return centroid_dict, ok

    def _compute_cog(self, subject, timepoints):
        for tp in timepoints:
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': tp})
            cog_path = seg_file.path.replace('nii.gz', 'npy').replace('T1wdseg', 'cog')

            seg_proxy = nib.load(seg_file.path)
            data = np.array(seg_proxy.dataobj)
            aux = np.where(data>0)
            i, j, k = np.median(aux[0]), np.median(aux[1]), np.median(aux[2])
            ras_cog = seg_proxy.affine @ np.array([i, j, k, 1])
            T_cog = np.eye(4)
            T_cog[:3, -1] = -ras_cog[:3]
            np.save(cog_path, T_cog.astype('float32'))

    def _init_graph(self, subject, timepoints, def_dir, force_flag):

        # compute centroids
        centroids_dict, ok_dict = self._get_centroids(subject, timepoints)

        for tp in timepoints:
            cog_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': tp,
                                         'suffix': 'cog', 'extension': 'npy'})
            T_cog = np.load(cog_file.path)
            centroids_dict[tp] = T_cog @ np.concatenate([centroids_dict[tp], np.ones((1, centroids_dict[tp].shape[1]))])
            centroids_dict[tp] = centroids_dict[tp][:3]

        # pairwise registration
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
            output_filepath = join(def_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.npy')
            if not exists(output_filepath) or force_flag:
                self._register_timepoints([centroids_dict[tp_ref], centroids_dict[tp_flo]], output_filepath,
                                          ok_centr=(ok_dict[tp_ref] == 1) & (ok_dict[tp_flo] == 1))

    def _solve_graph(self, subject, timepoints, def_dir, **kwargs):
        R_log = USLR_Linear.init_st2_lineal(timepoints, def_dir)
        Tres = USLR_Linear.st2_lineal_pytorch(R_log, timepoints, verbose=False, **kwargs)

        if np.sum(np.isnan(Tres)) > 0:
            return {'exit_code': -1, 'message': '[error] Something went wrong in the rigid registration step.\n'}

        for it_tp, tp in enumerate(timepoints):
            extra_kwargs = {'session': tp, 'subject': subject}
            cog_file = self._get_data(**{**self.seg_entities, **extra_kwargs, 'suffix': 'cog', 'extension': 'npy'})
            filename = self.build_path({**extra_kwargs, **self.aff_graph_entities})

            affine_matrix = Tres[..., it_tp]
            T_cog = np.load(cog_file.path)

            output_filepath = join(DIR_PIPELINES[self.pipeline_dir], filename)
            create_dir(dirname(output_filepath))

            np.save(output_filepath, np.linalg.inv(T_cog) @ affine_matrix)

        return {'exit_code': 2, 'message': '[partly done] graph is already computed; template and etiv missing.\n'}

    def _create_subject_space(self, subject, timepoints):
        # load segs, binarize, dilate and crop with 5 voxels per side.
        aff = {}
        masks = {}
        masks_dilated = {}
        orig_v2r = {}
        for tp in timepoints:
            filename = self.build_path({'session': tp, 'subject': subject, **self.aff_graph_entities})
            m = np.load(join(DIR_PIPELINES[self.pipeline_dir], filename))
            seg_file = self._get_data(**{'session': tp, 'subject': subject, **self.seg_entities})
            if m is not None and seg_file is not None:
                if np.sum(np.isnan(m)) > 0:
                    return {'exit_code': -1, 'message': '[error] Something went wrong in the rigid registration step.\n'}

                aff[tp] = m

                proxyseg = nib.load(seg_file)
                orig_v2r[tp] = proxyseg.affine

                seg = np.array(proxyseg.dataobj)
                mask = (seg > 0) & (seg != 24)
                masks[tp] = nib.Nifti1Image(mask, np.linalg.inv(m) @ proxyseg.affine)

                mask_dilated = binary_dilation(mask, ball(3)).astype('uint8')
                masks_dilated[tp] = nib.Nifti1Image(mask_dilated, np.linalg.inv(m) @ proxyseg.affine)

        # create subject space
        rasMosaic, template_vox2ras0, template_size = create_empty_template(list(masks_dilated.values()))
        save_nii(np.zeros(template_size), template_vox2ras0, join(self.tmp_dir, subject + '_template.nii.gz'))

        # move subject space to network space
        template = sf.load_volume(join(self.tmp_dir, subject + '_template.nii.gz'))
        net2vox, vox2net, net_v2r = network_space(template, shape=self.net_shape, center=template)
        proxytemplate = nib.Nifti1Image(np.zeros(self.net_shape), net_v2r)

        filename_ssspace = self.build_path({'subject': subject, **self.template_lin_entities})
        filename_ssseg = self.build_path({'subject': subject, **self.template_lin_entities, 'suffix': 'T1wdseg'})
        filename_ssmask = self.build_path({'subject': subject, **self.template_lin_entities, 'suffix': 'T1wmask'})
        filename_t_v2r = self.build_path({'subject': subject, **self.net_v2r_entities})

        create_dir(dirname(join(DIR_PIPELINES[self.pipeline_dir], filename_t_v2r)))
        np.save(join(DIR_PIPELINES[self.pipeline_dir], filename_t_v2r), net_v2r)
        os.remove(join(self.tmp_dir, subject + '_template.nii.gz'))

        # resample each timepoint to network space (images and dilated masks)
        image_list = []
        seg_list = []
        for tp in timepoints:
            proxymask = vol_resample_fast(proxytemplate, masks[tp])

            image_file = self._get_data(**{'subject': subject, 'session': tp, **self.bf_entities})
            if image_file is None:
                continue

            proxyraw = nib.load(image_file.path)
            pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
            new_vox_size = np.array([1, 1, 1])
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_array = np.array(proxyraw.dataobj)
            im_array = gaussian_filter(im_array, sigmas)
            proxyraw = nib.Nifti1Image(im_array, np.linalg.inv(aff[tp]) @ proxyraw.affine)
            proxyraw = vol_resample_fast(proxytemplate, proxyraw)
            image_list.append(proxyraw)

            seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
            proxyseg = nib.load(seg_file.path)
            arrayseg = np.array(proxyseg.dataobj)
            proxyseg = nib.Nifti1Image(arrayseg, np.linalg.inv(aff[tp]) @ proxyseg.affine)
            proxyseg = vol_resample_fast(proxytemplate, proxyseg, mode='nearest')
            seg_list.append(proxyseg)
            # arrayseg = np.array(proxyseg.dataobj)
            # arrayonehot = one_hot_encoding(arrayseg, categories=SYNTHSEG_APARC_LUT).astype('float')
            # proxyonehot = nib.Nifti1Image(arrayonehot, np.linalg.inv(aff[tp]) @ proxyseg.affine)
            # proxyonehot = vol_resample_fast(proxytemplate, proxyonehot)
            # proxyonehot.uncache()
            # seg_list.append(proxyonehot)

            # saving
            extra_kwargs = {'subject': subject, 'session': tp}
            filename_mask = self.build_path({**extra_kwargs, **self.mask_graph_lin_entities})
            filename_im = self.build_path({**extra_kwargs, **self.im_graph_lin_entities})
            filename_seg = self.build_path({**extra_kwargs, **self.im_graph_lin_entities, 'suffix': 'T1wdseg'})

            nib.save(proxymask, join(DIR_PIPELINES[self.pipeline_dir], filename_mask))
            nib.save(proxyraw, join(DIR_PIPELINES[self.pipeline_dir], filename_im))
            nib.save(proxyseg, join(DIR_PIPELINES[self.pipeline_dir], filename_seg))

        temp_array = np.stack([np.array(x.dataobj) for x in image_list], axis=0)
        temp_array = np.median(temp_array, axis=0)
        save_nii(temp_array, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], filename_ssspace))

        template_seg = np.zeros(proxytemplate.shape + (len(SYNTHSEG_APARC_LUT),))
        for proxyseg in seg_list:
            template_seg += one_hot_encoding(np.array(proxyseg.dataobj), categories=SYNTHSEG_APARC_LUT).astype('float')
            # template_seg += np.array(proxyseg.dataobj)

        template_seg = np.argmax(template_seg, axis=-1)
        template_seg = self._undo_one_hot(template_seg)
        template_mask = (template_seg > 0) & (template_seg != 24)

        save_nii(template_seg, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], filename_ssseg))
        save_nii(template_mask, net_v2r, join(DIR_PIPELINES[self.pipeline_dir], filename_ssmask))

        return {'exit_code': 3, 'message': '[partly done] graph and template are already computed; subject etiv missing.\n'}

    def _register_to_MNI(self, subject, timepoints, model='linear', **kwargs):

        mni_entities = {'subject': subject, 'space': 'MNI', 'desc': 'tosubject'}
        aff_fname = self.build_path({'suffix': 'aff', 'extension': 'npy', **mni_entities})
        svf_fname = self.build_path({'suffix': 'svf', 'extension': 'nii.gz', **mni_entities})
        v2r_fname = self.build_path({'suffix': 'v2r', 'extension': 'npy', **mni_entities})

        template_seg = self._get_data(**{**self.template_lin_entities,
                                         'subject': subject, 'suffix': 'T1wdseg', 'session': None})
        if template_seg is None:
            return

        centroid_ref, ok_ref = compute_centroids_ras(MNI_TEMPLATE_SEG, labels_registration)
        centroid_flo, ok_flo = compute_centroids_ras(template_seg.path, labels_registration)

        M_sbj = getM(centroid_ref[:, ok_ref > 0], centroid_flo[:, ok_ref > 0], use_L1=False)
        np.save(join(DIR_PIPELINES['lin'], aff_fname), M_sbj)

        proxytemplate = nib.load(MNI_TEMPLATE)
        if model == 'linear':
            proxyflow = None

        else:
            template_im = self._get_data(**{**self.template_lin_entities, 'subject': subject, 'session': None})
            if template_seg is None:
                return

            sfmni = sf.load_volume(MNI_TEMPLATE)
            net2vox, vox2net, net_v2r = network_space(sfmni, shape=self.net_shape, center=sfmni)
            np.save(join(DIR_PIPELINES['lin'], v2r_fname), net_v2r)
            svf_v2r = net_v2r.copy()
            for c in range(3):
                svf_v2r[:-1, c] = svf_v2r[:-1, c] / 0.5
            svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5] * 3) - 1))

            proxynet = nib.Nifti1Image(np.zeros(self.net_shape, dtype='float32'), net_v2r)
            proxytemplate = vol_resample_fast(proxynet, proxytemplate)

            proxysubject = nib.load(template_im)
            arrsubject = np.array(proxysubject.dataobj)
            proxysubject = nib.Nifti1Image(arrsubject, np.linalg.inv(M_sbj) @ proxysubject.affine)
            proxysubject = vol_resample_fast(proxynet, proxysubject)

            fw_svf = synthmorph_register(proxytemplate, proxysubject, reg_param=0.4)
            save_nii(fw_svf, svf_v2r, join(DIR_PIPELINES['lin'], svf_fname))

            flow = integrate_svf(fw_svf, self.net_shape, scaling_factor=2, int_steps=7)
            proxyflow = nib.Nifti1Image(flow, net_v2r)

        for tp in timepoints:
            image_file = self._get_data(**{'subject': subject, 'session': tp, **self.bf_entities})
            aff_file = self._get_data(**{'subject': subject, 'session': tp, **self.aff_graph_entities})
            seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
            if aff_file is None or image_file is None or seg_file is None:
                continue

            file_entities = {k: str(v) for k, v in image_file.entities.items() if k in filename_entities}
            if 'acquisition' in file_entities.keys():
                file_entities.pop('acquisition')

            file_entities['space'] = 'MNI'
            im_MNI_filepath = join(DIR_PIPELINES['lin'], self.build_path(file_entities))

            file_entities['suffix'] = 'T1wdseg'
            seg_MNI_filepath = join(DIR_PIPELINES['lin'], self.build_path(file_entities))

            aff = np.load(aff_file.path)

            proxyraw = nib.load(image_file.path)
            pixdim = np.sqrt(np.sum(proxyraw.affine * proxyraw.affine, axis=0))[:-1]
            new_vox_size = np.array([1, 1, 1])
            factor = pixdim / new_vox_size
            sigmas = 0.25 / factor
            sigmas[factor > 1] = 0  # don't blur if upsampling

            im_array = np.array(proxyraw.dataobj)
            im_array = gaussian_filter(im_array, sigmas)
            proxyraw = nib.Nifti1Image(im_array, np.linalg.inv(M_sbj) @ np.linalg.inv(aff) @ proxyraw.affine)
            proxyraw = vol_resample_fast(proxytemplate, proxyraw, proxyflow=proxyflow)
            nib.save(proxyraw, im_MNI_filepath)

            proxyseg = nib.load(seg_file.path)
            arrayseg = np.array(proxyseg.dataobj)
            proxyseg = nib.Nifti1Image(arrayseg, np.linalg.inv(M_sbj) @ np.linalg.inv(aff) @ proxyseg.affine)
            proxyseg = vol_resample_fast(proxytemplate, proxyseg, proxyflow=proxyflow, mode='nearest')
            nib.save(proxyseg, seg_MNI_filepath)

            # arrayseg = np.array(proxyseg.dataobj)
            # arrayonehot = one_hot_encoding(arrayseg, categories=SYNTHSEG_APARC_LUT).astype('float')
            # proxyonehot = nib.Nifti1Image(arrayonehot, np.linalg.inv(M_sbj) @ np.linalg.inv(aff) @ proxyseg.affine)
            # template_seg = np.array(proxyonehot.dataobj)
            # template_seg = np.argmax(template_seg, axis=-1)
            # template_seg = self._undo_one_hot(template_seg)
            # save_nii(template_seg.astype('uint8'), proxytemplate.affine, seg_MNI_filepath)

    def _compute_etiv(self, subject, timepoints):
        net_v2r = np.load(self._get_data(**{'subject': subject, **self.net_v2r_entities}).path)
        proxytemplate = nib.Nifti1Image(np.zeros(self.net_shape), net_v2r)
        template_mask = np.zeros(self.net_shape)
        for tp in timepoints:
            filename = self.build_path({'session': tp, 'subject': subject, **self.aff_graph_entities})
            aff = np.load(join(DIR_PIPELINES[self.pipeline_dir], filename))
            seg_file = self._get_data(**{'session': tp, 'subject': subject, **self.seg_entities})
            if aff is not None and seg_file is not None:
                proxyseg = nib.load(seg_file)
                arrseg = np.array(proxyseg.dataobj)
                arrmask = (arrseg > 0)

                proxyseg = nib.Nifti1Image(arrmask.astype('float'), np.linalg.inv(aff) @ proxyseg.affine)
                arrmask = vol_resample_fast(proxytemplate, proxyseg, return_np=True)
                template_mask += arrmask / len(timepoints)

        etiv = np.sum(template_mask)
        np.save(join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_T1wetiv.npy'), etiv)

    def process_subject(self, subject, force_flag=False, register_MNI=False, **kwargs):

        print('* Subject: ' + subject)
        def_dir = join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'deformations')
        create_dir(def_dir)

        timepoints = self._get_timepoints(subject=subject)
        checkpoint = self._check_running_subject(subject, timepoints, force_flag, register_MNI=register_MNI)
        print(checkpoint['message'])

        if checkpoint['exit_code'] == -1 or checkpoint['exit_code'] == 1:
            return

        if checkpoint['exit_code'] in [5]:
            # only one timepoint, check if needs to be registered to MNI.
            tp = timepoints[0]
            extra_kwargs = {'session': tp, 'subject': subject}
            im_fname = self.build_path({'subject': subject, **self.template_lin_entities, 'suffix': 'T1w'})
            im_temp_fpath = join(DIR_PIPELINES[self.pipeline_dir], im_fname)
            seg_fname = self.build_path({'subject': subject, **self.template_lin_entities, 'suffix': 'T1wdseg'})
            seg_temp_fpath = join(DIR_PIPELINES[self.pipeline_dir], seg_fname)

            im_fname = self.build_path({**extra_kwargs, **self.template_lin_entities, 'suffix': 'T1w'})
            im_fpath = join(DIR_PIPELINES[self.pipeline_dir], im_fname)
            seg_fname = self.build_path({**extra_kwargs, **self.template_lin_entities, 'suffix': 'T1wdseg'})
            seg_fpath = join(DIR_PIPELINES[self.pipeline_dir], seg_fname)


            aff_fname = self.build_path({**extra_kwargs, **self.aff_graph_entities})
            aff_fpath = join(DIR_PIPELINES[self.pipeline_dir], aff_fname)

            if not exists(aff_fpath):
                if not exists(dirname(aff_fpath)): makedirs(dirname(aff_fpath))
                np.save(aff_fpath, np.eye(4))

            if not exists(seg_fpath):
                if not exists(dirname(seg_fpath)): makedirs(dirname(seg_fpath))
                seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
                if seg_file is None:
                    return

                subprocess.call(['ln', '-s', seg_file.path, seg_fpath], stderr=subprocess.PIPE)
                # proxy = nib.load(seg_fpath)
                # seg = np.array(proxy.dataobj)
                # etiv = np.sum(seg > 0)
                # sid = 'sub-' + str(subject)
                # np.save(join(DIR_PIPELINES[self.pipeline_dir], sid, sid + '_T1wetiv.npy'), etiv)

            if not exists(im_fpath):
                if not exists(dirname(im_fpath)): makedirs(dirname(im_fpath))
                image_file = self._get_data(**{'subject': subject, 'session': tp, **self.bf_entities})
                if image_file is None:
                    return

                proxyim = nib.load(image_file.path)
                pixdim = np.sqrt(np.sum(proxyim.affine * proxyim.affine, axis=0))[:-1]
                if all([np.abs(p - 1) < 0.01 for p in pixdim]):
                    rf = subprocess.call(['ln', '-s', image_file.path, im_fpath], stderr=subprocess.PIPE)
                    if rf != 0:
                        subprocess.call(['cp', image_file.path, im_fpath])

                else:
                    # some dimension may be wrong
                    if any([p < 0.01 for p in pixdim]):
                        return

                    v, aff = rescale_voxel_size(np.array(proxyim.dataobj), proxyim.affine, [1, 1, 1])
                    save_nii(v.astype('float32'), aff, im_fpath)

            if not exists(seg_temp_fpath):
                if not exists(dirname(seg_temp_fpath)): makedirs(dirname(seg_temp_fpath))
                seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
                if seg_file is None:
                    return

                subprocess.call(['ln', '-s', seg_file.path, seg_temp_fpath], stderr=subprocess.PIPE)

            if not exists(im_temp_fpath):
                subprocess.call(['ln', '-s', im_fpath, im_temp_fpath], stderr=subprocess.PIPE)

            self._update_subject_layout(subject)

        if checkpoint['exit_code'] in [0]:
            # center images (cog)
            self._compute_cog(subject, timepoints)
            self._update_subject_layout(subject)

            # initialize graph
            self._init_graph(subject, timepoints, def_dir, force_flag)
            self._update_subject_layout(subject)

            # compute graph
            tmp_dir = join(self.tmp_dir, subject)
            create_dir(tmp_dir)
            graph_kwargs = {'n_epochs': 30, 'cost': 'l1', 'lr': 0.1, 'dir_results': tmp_dir, 'max_iter': 20}
            self._solve_graph(subject, timepoints, def_dir, **graph_kwargs)
            self._update_subject_layout(subject)

        if checkpoint['exit_code'] in [0, 2]:
            # create subject space
            checkpoint = self._create_subject_space(subject, timepoints)
            self._update_subject_layout(subject)

        if checkpoint['exit_code'] in [0, 2, 3]:
            # compute etiv
            self._compute_etiv(subject, timepoints)

        # register to MNI
        if register_MNI and checkpoint['exit_code'] in [0, 2, 3, 4, 5]:
            self._register_to_MNI(subject, timepoints, model='linear')
            self._update_subject_layout(subject)

        print('DONE. \n')

    @staticmethod
    def st2_lineal_pytorch(logR, timepoints, n_epochs, cost, lr, dir_results, max_iter=5, patience=3,
                           device='cpu', verbose=True):

        if len(timepoints) > 2:
            log_keys = ['loss', 'time_duration (s)']
            logger = History(log_keys)
            model_checkpoint = ModelCheckpoint(join(dir_results, 'checkpoints'), -1)
            callbacks = [logger, model_checkpoint]
            if verbose: callbacks += [PrinterCallback()]

            model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
            optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter,
                                          line_search_fn='strong_wolfe')

            min_loss = 1000
            iter_break = 0
            log_dict = {}
            logR = torch.FloatTensor(logR)
            for cb in callbacks:
                cb.on_train_init(model)

            for epoch in range(n_epochs):
                for cb in callbacks:
                    cb.on_epoch_init(model, epoch)

                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()

                    loss = model(logR, timepoints)
                    loss.backward()

                    return loss

                optimizer.step(closure=closure)

                loss = model(logR, timepoints)

                if loss < min_loss + 1e-4:
                    iter_break = 0
                    min_loss = loss.item()

                else:
                    iter_break += 1

                if iter_break > patience or loss.item() == 0.:
                    break

                log_dict['loss'] = loss.item()

                for cb in callbacks:
                    cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

            T = model._compute_matrix().cpu().detach().numpy()

        else:
            logR = np.squeeze(logR.astype('float32'))
            model = InstanceRigidModelLOG(timepoints, cost=cost, device=device, reg_weight=0)
            model.angle = nn.Parameter(torch.tensor(np.array([[-logR[0] / 2, logR[0] / 2],
                                                              [-logR[1] / 2, logR[1] / 2],
                                                              [-logR[2] / 2, logR[2] / 2]])).float(),
                                       requires_grad=False)

            model.translation = nn.Parameter(torch.tensor(np.array([[-logR[3] / 2, logR[3] / 2],
                                                                    [-logR[4] / 2, logR[4] / 2],
                                                                    [-logR[5] / 2, logR[5] / 2]])).float(),
                                             requires_grad=False)
            T = model._compute_matrix().cpu().detach().numpy()

        return T

    @staticmethod
    def init_st2_lineal(timepoints, input_dir, eps=1e-6):
        nk = 0

        N = len(timepoints)
        K = int(N * (N - 1) / 2)

        phi_log = np.zeros((6, K))

        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
            if not isinstance(tp_ref, str):
                tid_ref, tid_flo = tp_ref.id, tp_flo.id
            else:
                tid_ref, tid_flo = tp_ref, tp_flo

            filename = str(tid_ref) + '_to_' + str(tid_flo)

            rigid_matrix = np.load(join(input_dir, filename + '.npy'))
            rotation_matrix, translation_vector = rigid_matrix[:3, :3], rigid_matrix[:3, 3]

            # Log(R) and Log(T)
            t_norm = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1 + eps, 1 - eps)) + eps
            W = 1 / (2 * np.sin(t_norm)) * (rotation_matrix - rotation_matrix.T) * t_norm
            Vinv = np.eye(3) - 0.5 * W + ((1 - (t_norm * np.cos(t_norm / 2)) / (
                        2 * np.sin(t_norm / 2))) / t_norm ** 2) * W * W  # np.matmul(W, W)

            phi_log[0, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * t_norm
            phi_log[1, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * t_norm
            phi_log[2, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * t_norm

            phi_log[3:, nk] = np.matmul(Vinv, translation_vector)

            nk += 1

        return phi_log


class USLR_Deformable(USLRProcessing):

    @staticmethod
    def init_st2(timepoints, input_dir, image_shape, factor=1, mask_path=None, se=None, penalty=1, dict_flag=False):
        timepoints_dict = {t: it_t for it_t, t in enumerate(timepoints)}

        N = len(timepoints)
        K = int(N * (N - 1) / 2) + 1
        w = np.zeros((K, N), dtype='int')

        if dict_flag:
            obs_mask = {}
            phi = {}

        else:
            obs_mask = np.zeros(image_shape + (K,))
            phi = np.zeros(image_shape + (3, K,))

        nk = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
            t0 = timepoints_dict[tp_ref]
            t1 = timepoints_dict[tp_flo]
            filename = str(tp_ref) + '_to_' + str(tp_flo)

            proxysvf = nib.load(join(input_dir, filename + '.nii.gz'))
            arrsvf = np.asarray(proxysvf.dataobj)

            # Masks
            if mask_path is not None:
                mask_proxy = nib.load(mask_path)
                mask_proxy = vol_resample_fast(proxysvf, mask_proxy)
                mask = np.array(mask_proxy.dataobj)

            else:
                mask = np.ones(image_shape)

            if se is not None:
                mask = binary_dilation(mask, se)

            if dict_flag:
                phi[filename] = factor * arrsvf
                obs_mask[filename] = mask
            else:
                phi[..., nk] = factor * arrsvf
                obs_mask[..., nk] = mask

            w[nk, t0] = -1
            w[nk, t1] = 1
            nk += 1

        if not dict_flag:
            obs_mask[..., nk] = (np.sum(obs_mask[..., :nk - 1]) > 0).astype('uint8')

        w[nk, :] = penalty
        nk += 1
        return phi, obs_mask, w, nk

    @staticmethod
    def st2_L2_global(phi, W, N):
        precision = 1e-6
        lambda_control = np.linalg.inv((W.T @ W) + precision * np.eye(N)) @ W.T
        Tres = lambda_control @ np.transpose(phi, [0, 1, 2, 4, 3])
        Tres = np.transpose(Tres, [0, 1, 2, 4, 3])

        return Tres

    @staticmethod
    def st2_L1(phi, obs_mask, w, N, chunk_id=None, verbose=True):

        if chunk_id is not None and verbose:
            print("Processing chunk " + str(chunk_id))

        image_shape = obs_mask.shape[:3]
        Tres = np.zeros(image_shape + (3, N))

        for it_control_row in range(image_shape[0]):
            if np.mod(it_control_row, 10) == 0 and chunk_id is None and verbose:
                print('  * Row ' + str(it_control_row) + '/' + str(image_shape[0]))

            for it_control_col in range(image_shape[1]):
                for it_control_depth in range(image_shape[2]):
                    index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]

                    if index_obs.shape[0] > 0:
                        w_control = w[index_obs]
                        phi_control = phi[it_control_row, it_control_col, it_control_depth]
                        phi_control = phi_control[..., index_obs]
                        n_control = len(index_obs)

                        for it_dim in range(3):
                            # Set objective
                            c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)

                            # Set the inequality
                            A_lp = np.zeros((2 * n_control, n_control + N))
                            A_lp[:n_control, :n_control] = -np.eye(n_control)
                            A_lp[:n_control, n_control:] = -w_control
                            A_lp[n_control:, :n_control] = -np.eye(n_control)
                            A_lp[n_control:, n_control:] = w_control

                            reg = np.reshape(phi_control[it_dim], (n_control,))
                            b_lp = np.concatenate((-reg, reg), axis=0)

                            result = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=(None, None), method='highs-ds')
                            Tres[it_control_row, it_control_col, it_control_depth, it_dim] = result.x[n_control:]

        return Tres

    @staticmethod
    def st2_L1_chunks(phi, obs_mask, w, N, num_chunks=2, num_cores=4):
        if num_cores == 1:
            Tres = USLR_Deformable.st2_L1(phi, obs_mask, w, N)

        else:
            chunk_list = []
            image_shape = obs_mask.shape[:3]
            chunk_size = [int(np.ceil(cs / num_chunks)) for cs in image_shape]
            for x in range(num_chunks):
                for y in range(num_chunks):
                    for z in range(num_chunks):
                        max_x = min((x + 1) * chunk_size[0], image_shape[0])
                        max_y = min((y + 1) * chunk_size[1], image_shape[1])
                        max_z = min((z + 1) * chunk_size[2], image_shape[2])
                        chunk_list += [[slice(x * chunk_size[0], max_x),
                                        slice(y * chunk_size[1], max_y),
                                        slice(z * chunk_size[2], max_z)]]

            results = Parallel(n_jobs=num_cores)(
                delayed(USLR_Deformable.st2_L1)(
                    phi[chunk[0], chunk[1], chunk[2]], obs_mask[chunk[0], chunk[1], chunk[2]],
                    w, N, chunk_id=it_chunk) for it_chunk, chunk in enumerate(chunk_list))

            Tres = np.zeros(phi.shape[:4] + (N,))
            for it_chunk, chunk in enumerate(chunk_list):
                Tres[chunk[0], chunk[1], chunk[2]] = results[it_chunk]

        return Tres

    def _name(self):
        return 'USLR-NonLinearRegistration'

    def _build_processor(self):
        super()._build_processor()
        self.tmp_dir = join(self.tmp_dir, 'USLR_NonLin')
        create_dir(self.tmp_dir)


    def _init_graph(self, subject, timepoints, def_dir, force_flag=False):

        svf_v2r = np.load(self._get_data(**{'subject': subject, **self.svf_v2r_entities}).path)
        for tp_ref, tp_flo in itertools.permutations(timepoints, 2):
            output_filepath = join(def_dir, str(tp_ref) + '_to_' + str(tp_flo) + '.nii.gz')
            if exists(output_filepath) and not force_flag:
                continue

            # read image and mask
            imageref_file = self._get_data(**{'subject': subject, 'session': tp_ref, **self.im_graph_lin_entities})
            imageflo_file = self._get_data(**{'subject': subject, 'session': tp_flo, **self.im_graph_lin_entities})

            maskref_file = self._get_data(**{'subject': subject, 'session': tp_ref, **self.mask_graph_lin_entities})
            maskflo_file = self._get_data(**{'subject': subject, 'session': tp_flo, **self.mask_graph_lin_entities})

            if imageref_file is None or imageflo_file is None or maskref_file is None or maskflo_file is None:
                continue

            fw_svf = synthmorph_register(imageref_file, imageflo_file)

            img = nib.Nifti1Image(fw_svf, svf_v2r)
            nib.save(img, output_filepath)

    def _solve_graph(self, subject, timepoints, def_dir, cost, **kwargs):

        R, M, W, NK = USLR_Deformable.init_st2(timepoints, def_dir, self.svf_shape, se=None, dict_flag=False)

        if cost == 'bch-l2':
            T_latent = USLR_Deformable.st2_L2_global(R, W, len(timepoints))
            T_latent = {t: T_latent[..., it_t] for it_t, t in enumerate(timepoints)}

        else:
            T_latent = USLR_Deformable.st2_L1_chunks(R, M, W, len(timepoints), num_cores=1)
            T_latent = {t: T_latent[..., it_t] for it_t, t in enumerate(timepoints)}

        return T_latent

    def _compute_template(self, subject, timepoints, cost, **kwargs):
        sss_file = self._get_data(**{'subject': subject, **self.template_lin_entities})
        if sss_file is None:
            return

        proxyref = nib.load(sss_file.path)

        # build path template: image, mask, seg
        image_filename = self.build_path({'suffix': 'T1w', 'subject': subject, **self.template_nonlin_entities})
        image_std_filename = self.build_path({'suffix': 'T1wstd', 'subject': subject, **self.template_nonlin_entities})
        mask_filename = self.build_path({'suffix': 'T1wmask', 'subject': subject, **self.template_nonlin_entities})
        seg_filename = self.build_path({'suffix': 'T1wdseg', 'subject': subject, **self.template_nonlin_entities})

        # compute template: image, mask, seg
        image_list = []
        seg_list = []
        for tp in timepoints:
            im_file = self._get_data(**{'subject': subject, 'session': tp, **self.bf_entities})
            seg_file = self._get_data(**{'subject': subject, 'session': tp, **self.seg_entities})
            aff_file = self._get_data(**{'subject': subject, 'session': tp, **self.aff_graph_entities})
            svf_file = self._get_data(**{'subject': subject, 'session': tp, **self.svf_graph_entities})

            if svf_file is None or aff_file is None or im_file is None or seg_file is None:
                continue

            proxyimage = nib.load(im_file.path)
            proxyseg = nib.load(seg_file.path)
            aff = np.load(aff_file.path)
            proxysvf = nib.load(svf_file.path)
            flow = integrate_svf(np.array(proxysvf.dataobj), self.net_shape, scaling_factor=2, int_steps=7)
            proxyflow = nib.Nifti1Image(flow, affine=proxyref.affine)

            # Image
            arrayim = np.array(proxyimage.dataobj)
            arrayim = gaussian_antialiasing(arrayim, proxyimage.affine, [1, 1, 1])
            proxyimage = nib.Nifti1Image(arrayim, np.linalg.inv(aff) @ proxyimage.affine)
            proxyimage = vol_resample_fast(proxyref, proxyimage, proxyflow=proxyflow)
            proxyimage.uncache()

            arrayseg = np.array(proxyseg.dataobj)
            arrayonehot = one_hot_encoding(arrayseg, categories=SYNTHSEG_APARC_LUT).astype('float')
            proxyonehot = nib.Nifti1Image(arrayonehot, np.linalg.inv(aff) @ proxyseg.affine)
            proxyonehot = vol_resample_fast(proxyref, proxyonehot, proxyflow=proxyflow)
            proxyonehot.uncache()

            image_list.append(proxyimage)
            seg_list.append(proxyonehot)

        # save image (median and std), mask (and etiv) and seg.
        arr_image_list = np.stack([np.array(proxyim.dataobj) for proxyim in image_list], axis=0)
        template = np.median(arr_image_list, axis=0)
        template_std = np.std(arr_image_list, axis=0)
        del arr_image_list

        template_seg = np.zeros(template.shape + (len(SYNTHSEG_APARC_LUT),))
        for proxyseg in seg_list:
            template_seg += np.array(proxyseg.dataobj)

        template_seg = np.argmax(template_seg, axis=-1)
        template_seg = self._undo_one_hot(template_seg)
        template_mask = (template_seg > 0) & (template_seg != 24)
        etiv = np.sum(template_seg > 0)

        save_nii(template, proxyref.affine, join(DIR_PIPELINES['nonlin-' + cost], image_filename))
        save_nii(template_std, proxyref.affine, join(DIR_PIPELINES['nonlin-' + cost], image_std_filename))
        save_nii(template_seg, proxyref.affine, join(DIR_PIPELINES['nonlin-' + cost], seg_filename))
        save_nii(template_mask, proxyref.affine, join(DIR_PIPELINES['nonlin-' + cost], mask_filename))

        sid = 'sub-' + str(subject)
        np.save(join(DIR_PIPELINES['nonlin-' + cost], sid, sid + '_T1wetiv.npy'), etiv)

    def _compute_mean_svf(self, subject, timepoints, cost, **kwargs):

        from sklearn.linear_model import LinearRegression

        fit_entities = {'subject': subject, 'space': 'uslrnonlin', 'task': 'linfit', 'extension': 'nii.gz',
                        'scope': 'nonlin-' + cost, 'suffix': 'svf'}
        image_filename = self.build_path({'desc': 'mean', **fit_entities})
        error_filename = self.build_path({'desc': 'error', **fit_entities})
        std_filename = self.build_path({'desc': 'std', **fit_entities})

        fit_entities['suffix'] = 'def'
        error_flow_filename = self.build_path({'desc': 'error', **fit_entities})
        std_flow_filename = self.build_path({'desc': 'std', **fit_entities})

        fit_entities['suffix'] = 'jac'
        jac_filename = self.build_path({'desc': 'mean', **fit_entities})

        linreg = LinearRegression()
        time_list = self._get_time_list(subject, timepoints)

        file_v2r = self._get_data(**{'subject': subject, **self.svf_v2r_entities})
        if file_v2r is None:
            return

        svf_v2r = np.load(file_v2r.path)

        svf_list = []
        features_list = []
        for tp in timepoints:
            svf_file = self._get_data(**{'subject': subject, 'session': tp, **self.svf_graph_entities})
            if svf_file is None:
                continue

            proxysvf = nib.load(svf_file.path)
            svf_list.append(np.array(proxysvf.dataobj).reshape(-1))

            age = float(time_list[tp])
            features_list.append([age])

        X = np.array(features_list)
        Y = np.stack(svf_list, axis=0)
        linreg.fit(X, Y)

        coef_list = [linreg.coef_[:, it_f].reshape(self.svf_shape + (3,)) for it_f in range(len(features_list[0]))]
        intercept_list = [linreg.intercept_.reshape(self.svf_shape + (3,))]
        results_vol = np.stack(intercept_list + coef_list, axis=-1)

        y_pred = linreg.predict(X)
        error_vol = np.sum((Y - y_pred) ** 2 / len(svf_list), axis=0).reshape(self.svf_shape + (3,))
        std_vol = np.sum((Y - np.mean(Y, axis=0)) ** 2 / len(svf_list), axis=0).reshape(self.svf_shape + (3,))

        save_nii(results_vol, svf_v2r, join(DIR_PIPELINES['nonlin-' + cost], image_filename))
        save_nii(error_vol, svf_v2r, join(DIR_PIPELINES['nonlin-' + cost], error_filename))
        save_nii(std_vol, svf_v2r, join(join(DIR_PIPELINES['nonlin-' + cost], std_filename)))

        # Flow error
        flow_pred = []
        flow = []
        for it_tp in range(Y.shape[0]):
            svf_pred = y_pred[it_tp].reshape(self.svf_shape + (3,))
            svf = Y[it_tp].reshape(self.svf_shape + (3,))
            flow_pred += [integrate_svf(svf_pred, self.net_shape, scaling_factor=2, int_steps=7)]
            flow += [integrate_svf(svf, self.net_shape, scaling_factor=2, int_steps=7)]

        flow_pred = np.stack(flow_pred, axis=0)
        flow = np.stack(flow, axis=0)
        flow_error_vol = np.abs(flow - flow_pred).reshape((len(flow),) + self.net_shape + (3,))
        flow_std_vol = (flow - np.mean(flow, axis=0) ** 2).reshape((len(flow),) + self.net_shape + (3,))

        save_nii(flow_error_vol, svf_v2r, join(DIR_PIPELINES['nonlin-' + cost], error_flow_filename))
        save_nii(flow_std_vol, svf_v2r, join(join(DIR_PIPELINES['nonlin-' + cost], std_flow_filename)))

        net_v2r = np.load(self._get_data(**{'subject': subject, **self.net_v2r_entities}).path)
        svf = results_vol[..., 1]
        if max(time_list.values()) - min(time_list.values()) > 30:
            svf = svf*365.25

        flow = integrate_svf(svf, self.net_shape, scaling_factor=2, int_steps=7)
        jac = compute_jacobian(flow)
        save_nii(jac, net_v2r, join(DIR_PIPELINES['nonlin-' + cost], jac_filename))


        seg_filename = self.build_path({'suffix': 'T1wdseg', 'subject': subject, **self.template_nonlin_entities})
        proxyseg = nib.load(join(DIR_PIPELINES['nonlin-' + cost], seg_filename))
        seg_cort = np.array(proxyseg.dataobj)

        proxyseg = nib.load(join(DIR_PIPELINES['lin'], seg_filename))
        seg_subcort = np.array(proxyseg.dataobj)

        filepath = join(DIR_PIPELINES['nonlin-' + cost], 'sub-' + str(subject), 'sub-' + str(subject) + '_jac.csv')

        if exists(filepath):
            data_df = pd.read_csv(filepath, dtype=str)
        else:
            data_df = pd.DataFrame(columns=['metric'] + list(self.labels_dict.values()))

        for mn, mf in {'mean': np.mean, 'median': np.median, 'std': np.std, 'min': np.min, 'max': np.max}.items():
            d = {'metric': [mn]}
            for lnum, lname in self.labels_dict.items():
                if 'ctx' in lname:
                    seg = seg_cort
                else:
                    seg = seg_subcort

                jval = jac[seg == lnum]
                if lnum in seg:
                    d[lname] = [mf(jval)]
                else:
                    d[lname] = [np.nan]

            data_df = pd.concat([data_df, pd.DataFrame(d)], ignore_index=True)

        data_df.to_csv(filepath, index=False)

    def _register_to_MNI(self, subject, cost, force_flag=False, **kwargs):
        scope = 'nonlin-' + cost
        jac_entities = {'subject': subject, 'task': 'linfit', 'scope': scope, 'suffix': 'jac', 'desc': 'mean'}
        mni_entities = {'subject': subject, 'space': 'MNI', 'desc': 'tosubject'}
        aff_fname = self.build_path({'suffix': 'aff', 'extension': 'npy', **mni_entities})
        svf_fname = self.build_path({'suffix': 'svf', 'extension': 'nii.gz', **mni_entities})
        v2r_fname = self.build_path({'suffix': 'v2r', 'extension': 'npy', **mni_entities})

        template_im = self._get_data(**{'suffix': 'T1w', 'subject': subject, **self.template_nonlin_entities})
        template_seg = self._get_data(**{'suffix': 'T1wdseg', 'subject': subject, **self.template_nonlin_entities})
        if template_seg is None or template_im is None:
            return

        centroid_ref, ok_ref = compute_centroids_ras(MNI_TEMPLATE_SEG, labels_registration)
        centroid_flo, ok_flo = compute_centroids_ras(template_seg.path, labels_registration)

        M_sbj = getM(centroid_ref[:, ok_ref > 0], centroid_flo[:, ok_ref > 0], use_L1=False)
        np.save(join(DIR_PIPELINES[scope], aff_fname), M_sbj)

        sfmni = sf.load_volume(MNI_TEMPLATE)
        net2vox, vox2net, net_v2r = network_space(sfmni, shape=self.net_shape, center=sfmni)
        np.save(join(DIR_PIPELINES[scope], v2r_fname), net_v2r)
        svf_v2r = net_v2r.copy()
        for c in range(3):
            svf_v2r[:-1, c] = svf_v2r[:-1, c] / 0.5
        svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5] * 3) - 1))

        proxynet = nib.Nifti1Image(np.zeros(self.net_shape, dtype='float32'), net_v2r)
        proxytemplate = nib.load(MNI_TEMPLATE)
        proxytemplate = vol_resample_fast(proxynet, proxytemplate)

        proxysubject = nib.load(template_im)
        arrsubject = np.array(proxysubject.dataobj)
        proxysubject = nib.Nifti1Image(arrsubject, np.linalg.inv(M_sbj) @ proxysubject.affine)
        proxysubject = vol_resample_fast(proxynet, proxysubject)

        fw_svf = synthmorph_register(proxytemplate, proxysubject, reg_param=0.4)
        save_nii(fw_svf, svf_v2r, join(DIR_PIPELINES['nonlin-' + cost], svf_fname))

        jac_entities['space'] = 'uslr'
        jac_file = self._get_data(**jac_entities)
        if jac_file is not None:
            proxyjac = nib.load(jac_file.path)
            arrjac = np.array(proxyjac.dataobj)
            proxyjac = nib.Nifti1Image(arrjac, np.linalg.inv(M_sbj) @ proxyjac.affine)
            flow = integrate_svf(fw_svf, self.net_shape, scaling_factor=2, int_steps=7)
            proxyflow = nib.Nifti1Image(flow, net_v2r)
            proxysubject = vol_resample_fast(proxynet, proxyjac, proxyflow=proxyflow)
            jac_entities['space'] = 'MNI'
            jac_filename = self.build_path(jac_entities)
            nib.save(proxysubject, join(DIR_PIPELINES[scope], jac_filename))

    def process_subject(self, subject, cost='bch-l1', force_flag=False, **kwargs):
        print('* Subject: ' + subject)
        assert cost in ['bch-l1', 'bch-l2', 'l1', 'l2']
        self.svf_graph_entities['scope'] = 'nonlin-' + cost

        timepoints = self._get_timepoints(subject=subject, uslr=False)

        def_dir = join(DIR_PIPELINES['nonlin'], 'sub-' + subject, 'deformations')
        create_dir(def_dir)

        # # compute svf v2r
        # svf_v2r_file = self._get_data(subject=subject, **self.svf_v2r_entities)
        # if svf_v2r_file is None:
        #     net_v2r = np.load(self._get_data(subject=subject, **self.net_v2r_entities).path)
        #     svf_v2r = net_v2r.copy()
        #     for c in range(3):
        #         svf_v2r[:-1, c] = svf_v2r[:-1, c] / 0.5
        #     svf_v2r[:-1, -1] = svf_v2r[:-1, -1] - np.matmul(svf_v2r[:-1, :-1], 0.5 * (np.array([0.5] * 3) - 1))
        #
        #     filename_v2r = self.build_path({'subject': subject, **self.svf_v2r_entities})
        #     np.save(join(DIR_PIPELINES['lin'], str(filename_v2r)), svf_v2r)
        #
        # else:
        #     svf_v2r = np.load(svf_v2r_file.path)
        #
        # # build the entire graph
        # self._init_graph(subject, timepoints, def_dir, force_flag)
        # self._update_subject_layout(subject)
        #
        # # solve spanning tree
        # T_latent = self._solve_graph(subject, timepoints, def_dir, cost, **kwargs)
        # for it_tp, tp in enumerate(timepoints):
        #     filename = self.build_path({'subject': subject, 'session': tp, **self.svf_graph_entities})
        #     create_dir(dirname(join(DIR_PIPELINES['nonlin-' + cost], filename)))
        #
        #     img = nib.Nifti1Image(T_latent[tp].astype('float32'), svf_v2r)
        #     nib.save(img, join(DIR_PIPELINES['nonlin-' + cost], filename))
        # self._update_subject_layout(subject)
        #
        # # create subject space
        # self._compute_template(subject, timepoints, cost)
        # self._update_subject_layout(subject)

        # compute mean SVF
        self._compute_mean_svf(subject, timepoints, cost)
        self._update_subject_layout(subject)

        # register to MNI
        # self._register_to_MNI(subject, cost)

        print('DONE. \n')


class USLR_Deformable_Exact(USLR_Deformable):
    '''
    It implements the USLR solution without BCH approximation
    '''

    @staticmethod
    def st2_nonlinear_pytorch(phi, obs_size, cp_size, n_epochs, cost, lr, tmp_dir, mask=None, patience=3,
                              init_T=None, timepoints=None, device='cpu', verbose=True, reg_weight=0.001):

        log_keys = ['loss', 'time_duration (s)']
        logger = History(log_keys)
        model_checkpoint = ModelCheckpoint(join(tmp_dir, 'checkpoints'), -1)
        callbacks = [logger, model_checkpoint]
        if verbose: callbacks += [PrinterCallback()]

        model = ST2Nonlinear(obs_size, cp_size, init_T=init_T, timepoints=timepoints, cost=cost, device=device,
                             reg_weight=reg_weight, version=0)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, betas=(0.5, 0.999))
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=10, line_search_fn='strong_wolfe',
                                      history_size=1)

        tid_list = list(phi.keys())
        S = torch.from_numpy(np.stack(list(phi.values()), axis=-1)).to(device)
        R = torch.zeros(cp_size + (3, S.shape[-1])).to(device)
        for it in range(S.shape[-1]):
            R[..., it] = torch.permute(
                model.integrate(torch.unsqueeze(torch.permute(S[..., it], dims=(3, 0, 1, 2)), 0))[0], dims=(1, 2, 3, 0))

        M = None if mask is None else torch.from_numpy(np.stack(list(mask.values()), axis=-1)).to(device)
        if M is not None:
            M = torch.permute(model.downscale(torch.unsqueeze(torch.permute(M, dims=(3, 0, 1, 2)), 0))[0],
                              dims=(1, 2, 3, 0))

        min_loss = 1e10
        iter_break = 0
        log_dict = {}
        for cb in callbacks:
            cb.on_train_init(model)

        model.train()
        for epoch in range(n_epochs):
            for cb in callbacks:
                cb.on_epoch_init(model, epoch)

            optimizer.zero_grad()

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                loss = model(R, tid_list, M=M)
                loss.backward()
                return loss

            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure=closure)
                loss = model(R, tid_list, M=M)

            else:
                loss = model(R, tid_list, M=M)
                loss.backward()
                optimizer.step()

            loss = loss.item()
            if ((min_loss - loss) / np.abs(min_loss)) * 100 > 0.1:
                print('Improving:', end=' ', flush=True)
                print(((min_loss - loss) / np.abs(min_loss)) * 100)

                iter_break = 0
                min_loss = loss

            else:
                print('Not improving:', end=' ', flush=True)
                print(((min_loss - loss) / np.abs(min_loss)) * 100)
                iter_break += 1

            if iter_break > patience or loss == 0.:
                break

            log_dict['loss'] = loss

            for cb in callbacks:
                cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

        # return {tid: model.upscale(model.T[tid]) for tid in model.T.keys()}
        return model.T

    def _solve_graph(self, subject, timepoints, def_dir, cost='l1', initialize_bch_l2=False):

        net_file = self._get_data(space='uslrlin', subject=subject, acquisition='1', suffix='space', extension='nii.gz')
        if net_file is None:
            return

        proxynet = nib.load(net_file.path)

        R, M, W, NK = USLR_Deformable.init_st2(timepoints, def_dir, self.svf_shape, se=None, dict_flag=False)

        # Initialization
        if initialize_bch_l2:
            Tres = USLR_Deformable.st2_L2_global(R, W, len(timepoints))
            T_latent = {t.id: torch.from_numpy(Tres[..., it_t][np.newaxis]) for it_t, t in enumerate(timepoints)}

        else:
            T_latent = None

        graph_structure = USLR_Deformable.init_st2(timepoints, def_dir, self.svf_shape, se=None, dict_flag=True)
        R, _, _, _ = graph_structure

        M = {}
        for k in R.keys():
            proxy = nib.load(join(def_dir, k.split('_to_')[0] + '.mask.nii.gz'))
            M[k] = vol_resample_fast(proxynet, proxy, return_np=True, mode='linear')

        if len(timepoints) > 2:
            T_latent = USLR_Deformable_Exact.st2_nonlinear_pytorch(
                R, self.net_shape, self.svf_shape, 50, cost=cost, lr=1, tmp_dir=self.tmp_dir, patience=5,
                init_T=T_latent, device='cuda:0', verbose=True, reg_weight=0.001, mask=M, timepoints=timepoints)

        T_latent = {k: v[0].detach().cpu().numpy() for k, v in T_latent.items()}

        return T_latent


class USLR_LabelFusion(GenerativeLabelFusionProcessing, USLRProcessing):

    @property
    def pipeline_dir(self):
        return self._pipeline_dir

    @pipeline_dir.setter
    def pipeline_dir(self, value):
        self._pipeline_dir = value

    def _name(self):
        return 'USLRLabelFusion (gaussian kernel)'

    def _build_processor(self):
        super()._build_processor()
        self.tmp_dir = join(self.tmp_dir, 'USLR-LF-' + str(self.pipeline_dir))
        create_dir(self.tmp_dir)

    def _get_tp_displacement(self, subject, target_tp, atlas_tp, **kwargs):
        seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
        aff_ref_file = self._get_data(**{'subject': subject, 'session': target_tp, **self.aff_graph_entities})
        svf_ref_file = self._get_data(**{'subject': subject, 'session': target_tp, **self.svf_graph_entities})
        if seg_ref_file is None or aff_ref_file is None or svf_ref_file is None:
            return

        proxysegref = nib.load(seg_ref_file.path)
        affref = np.linalg.inv(np.load(aff_ref_file.path))
        proxysvfref = nib.load(svf_ref_file.path)

        seg_flo_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})
        aff_flo_file = self._get_data(**{'subject': subject, 'session': atlas_tp, **self.aff_graph_entities})
        svf_flo_file = self._get_data(**{'subject': subject, 'session': atlas_tp, **self.svf_graph_entities})
        if seg_flo_file is None or aff_flo_file is None or svf_flo_file is None:
            return

        proxysegflo = nib.load(seg_flo_file.path)
        affflo = np.load(aff_flo_file.path)
        proxysvfflo = nib.load(svf_flo_file.path)

        file_flow_v2r = self._get_data(**{'subject': subject, **self.net_v2r_entities})
        if file_flow_v2r is None:
            return

        flow_v2r = np.load(file_flow_v2r.path)

        svf = np.array(proxysvfflo.dataobj) - np.array(proxysvfref.dataobj)
        flow = integrate_svf(svf, self.net_shape, scaling_factor=2, int_steps=7)

        return compose_transforms((tf.cast(np.linalg.inv(proxysegflo.affine) @ affflo @ flow_v2r, tf.float32), flow,
                                   tf.cast(np.linalg.inv(flow_v2r) @ affref @ proxysegref.affine, tf.float32)),
                                  shift_center=False, shape=proxysegref.shape)

    # def _deform_atlases(self, subject, target_tp, timepoints, results_dir, *args, **kwargs):
    #     seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
    #     if seg_ref_file is None:
    #         return
    #
    #     proxysegref = nib.load(seg_ref_file.path)
    #     for atlas_tp in timepoints:
    #         im_file = self._get_data(**{**self.bf_entities, 'subject': subject, 'session': atlas_tp})
    #         seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})
    #
    #         if im_file is None or seg_file is None:
    #             continue
    #
    #         output_image_filepath = join(results_dir, str(atlas_tp) + '.im.nii.gz')
    #         output_onehot_filepath = join(results_dir, str(atlas_tp) + '.onehot.nii.gz')
    #
    #         # One-hot encoding of the labels
    #         proxyseg = nib.load(seg_file.path)
    #         seg_arr = np.array(proxyseg.dataobj)
    #         onehot_arr = one_hot_encoding(seg_arr, categories=self.labels_lut)
    #
    #         # Gaussian filter for [1, 1, 1] segmentation
    #         proxyim = nib.load(im_file.path)
    #         arrayim = np.array(proxyim.dataobj)
    #         arrayim = gaussian_antialiasing(arrayim, proxyim.affine, [1, 1, 1])
    #         proxyim = nib.Nifti1Image(arrayim, proxyim.affine)
    #         proxyim = vol_resample_fast(proxyseg, proxyim)
    #         arrayim = np.array(proxyim.dataobj)
    #
    #         if target_tp == atlas_tp:
    #             proxyonehot = nib.Nifti1Image(onehot_arr.astype('float32'), proxyseg.affine)
    #             nib.save(proxyonehot, output_onehot_filepath)
    #             nib.save(proxyim, output_image_filepath)
    #
    #         else:
    #             tp_displ = self._get_tp_displacement(subject, target_tp, atlas_tp, **kwargs)
    #
    #             # Image
    #             arrayim_mov = warp(arrayim, tp_displ)
    #             proxyim = nib.Nifti1Image(arrayim_mov.astype('float32'), proxysegref.affine)
    #             nib.save(proxyim, output_image_filepath)
    #
    #             arrayonehot_mov = warp(onehot_arr, tp_displ)
    #             proxyonehot = nib.Nifti1Image(arrayonehot_mov.astype('float32'), proxysegref.affine)
    #             nib.save(proxyonehot, output_onehot_filepath)

    def process_subject(self, subject, cost='bch-l2', time_scale=None, g_std=None, **kwargs):
        assert cost in ['bch-l1', 'bch-l2', 'l1', 'l2', 'lin', 'lin-reverse']
        if cost not in ['lin', 'lin-reverse']:
            self.svf_graph_entities['scope'] = 'nonlin-' + cost

        super(USLR_LabelFusion, self).process_subject(subject, time_scale=time_scale, g_std=g_std, **kwargs)


class USLR_Lin_LabelFusion(USLR_LabelFusion):
    def _get_tp_displacement(self, subject, target_tp, atlas_tp, **kwargs):
        seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
        aff_ref_file = self._get_data(**{'subject': subject, 'session': target_tp, **self.aff_graph_entities})
        if seg_ref_file is None or aff_ref_file is None:
            return

        proxysegref = nib.load(seg_ref_file.path)
        affref = np.linalg.inv(np.load(aff_ref_file.path))

        seg_flo_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})
        aff_flo_file = self._get_data(**{'subject': subject, 'session': atlas_tp, **self.aff_graph_entities})
        if seg_flo_file is None or aff_flo_file is None:
            return

        proxysegflo = nib.load(seg_flo_file.path)
        affflo = np.load(aff_flo_file.path)

        flow = np.zeros(self.net_shape + (3,))
        return compose_transforms((tf.cast(np.linalg.inv(proxysegflo.affine) @ affflo, tf.float32), flow,
                                   tf.cast(affref @ proxysegref.affine, tf.float32)),
                                  shift_center=False, shape=proxysegref.shape)
