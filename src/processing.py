import pdb
from os.path import isfile, join, dirname, basename, exists
from os import makedirs
import copy
from joblib import delayed, Parallel

from bids.layout import BIDSLayout, BIDSLayoutIndexer
import numpy as np
import nibabel as nib
import pandas as pd

from utils.log_utils import LogBIDSLoader
from utils.label_utils import SYNTHSEG_APARC_LUT, SYNTHSEG_APARC_DICT
from utils.io_utils import create_dir, remove_dir
from utils.def_utils import vol_resample_fast
from utils.fn_utils import gaussian_antialiasing, one_hot_encoding
from utils.synthmorph_utils import warp
from setup import *

def parallel_processing(processing, subject, **kwargs):
    # processing = copy.deepcopy(self)
    processing._update_subject_layout(subject)
    processing.process_subject(subject, **kwargs)


class Processing(object):
    def __init__(self, bids_loader, subject_list=None, **kwargs):
        self.bids_loader = bids_loader
        self.subject_list = bids_loader.get_subjects() if subject_list is None else subject_list
        # self.subject_list = [s for s in self.subject_list if s not in ['MIRIAD189']]

        self.bids_logger = LogBIDSLoader(num_files=1)
        self._build_processor()
        create_dir(self.tmp_dir)

    def _build_processor(self):
        self.tmp_dir = TMP_DIR
        create_dir(self.tmp_dir)

        self.seg_entities = {'scope': 'preproc', 'extension': 'nii.gz', 'suffix': ['T1wdseg', 'dseg']}
        self.bf_entities = {'scope': 'preproc', 'extension': 'nii.gz', 'suffix': 'T1w', 'acquisition': [None, 'orig']}

        self.labels_lut = SYNTHSEG_APARC_LUT
        self.labels_dict = SYNTHSEG_APARC_DICT

    def build_path(self, entities, absolute_paths=False, validate=False, strict=True):
        entities = {k: v for k, v in entities.items() if k in filename_entities}
        return self.bids_loader.build_path(entities, absolute_paths=absolute_paths, path_patterns=BIDS_PATH_PATTERN,
                                           strict=strict, validate=validate)

    def _name(self):
        return ''

    def get_subjects(self, uslr=True):
        subjects = self.bids_loader.get_subjects()
        if uslr:
            subjects = list(filter(lambda s: len(self._get_data(**{'subject': s, **self.seg_entities},
                                                                ignore_check=True)) > 0 is not None, subjects))

        return subjects
    def _get_timepoints(self, subject, uslr=True):
        timepoints = self.bids_loader.get_session(subject=subject)
        if uslr:
            timepoints = list(filter(lambda tp: self._get_data(**{
                'session': tp, 'subject': subject, **self.seg_entities}, verbose=False) is not None, timepoints))

        return timepoints
    def _get_data(self, ignore_check=False, curr_len=None, verbose=True, **kwargs):
        file_list = self.bids_loader.get(**kwargs)
        if ignore_check:
            return file_list

        file_flag = self.bids_logger.check_length(file_list, curr_len=curr_len)
        if file_flag['exit_code'] == -1:
            if verbose:
                print('[warning]', end=' ', flush=True)
                print(file_flag['log'], end=' ', flush=True)
                print(' --> Entities: ' + ','.join(['<' + str(k) + ':' + str(v) + '>' for k, v in kwargs.items()]))
            raw_file = None
        else:
            raw_file = file_flag['file']

        return raw_file

    def _get_entities(self, file):
        return {k: v for k, v in file.entities.items() if k in filename_entities}

    def _on_pipeline_init(self):
        name = self._name()
        if len(name) > 0:
            print('\n\n\n\n\n')
            print('# ' + '-'.join([''] * (len(name) + 7)) + ' #')
            print('#    ' + name + '    #')
            print('# ' + '-'.join([''] * (len(name) + 7)) + ' #')
            print('\n\n')

    def _update_subject_layout(self, subject):
        rawdir = self.bids_loader.root
        derivatives = self.bids_loader.derivatives.keys()

        indexer = BIDSLayoutIndexer(validate=False, ignore='sub-(?!' + subject + ')(.*)$', index_metadata=False)
        bids_kwargs = {'validate': False, 'indexer': indexer}

        bids_loader = BIDSLayout(root=rawdir, **bids_kwargs)
        bids_loader.add_derivatives([DIR_PIPELINES[d] for d in derivatives], **bids_kwargs)

        self.bids_loader = bids_loader

    def _update_full_layout(self):
        rawdir = self.bids_loader.root
        derivatives = self.bids_loader.derivatives.keys()

        indexer = BIDSLayoutIndexer(validate=False, index_metadata=False)
        bids_kwargs = {'validate': False, 'indexer': indexer}

        bids_loader = BIDSLayout(root=rawdir, **bids_kwargs)
        bids_loader.add_derivatives([DIR_PIPELINES[d] for d in derivatives], **bids_kwargs)

        self.bids_loader = bids_loader

    def _get_subject_info(self, subject):
        sess_df = None
        sess_tsv = self._get_data(suffix='sessions', extension='tsv', subject=subject)
        if sess_tsv:
            sess_df = sess_tsv.get_df()
            sess_df = sess_df.set_index('session_id')
            sess_df = sess_df[~sess_df.index.duplicated(keep='last')]

        return sess_df

    def _get_participant_info(self):
        part_df = None
        part_tsv = self._get_data(suffix='participants', extension='tsv')
        if part_tsv:
            part_df = part_tsv.get_df()
            part_df = part_df.set_index('participant_id')
            part_df = part_df[~part_df.index.duplicated(keep='last')]

        return part_df

    def _get_time_list(self, subject, timepoints, sess_df=None):
        time_list = {tp: 0 for tp in timepoints}
        if sess_df is None:
            sess_df = self._get_subject_info(subject)

        if sess_df is not None:
            if 'time_to_bl_days' in sess_df.iloc[0].index:
                time_list = {tp: float(sess_df.loc[tp]['time_to_bl_days']) for tp in timepoints}

            elif 'time_to_bl_years' in sess_df.iloc[0].index:
                time_list = {tp: float(sess_df.loc[tp]['time_to_bl_years']) for tp in timepoints}

            elif 'age' in sess_df.iloc[0].index:
                time_list = {tp: float(sess_df.loc[tp]['age']) for tp in timepoints}

        return time_list

    def _get_last_tp(self, subject, timepoints, time_list=None):
        if time_list is None:
            time_list = self._get_time_list(subject, timepoints)

        tp_id = list(time_list.keys())
        tp_time = list(time_list.values())
        return tp_id[np.argmax(tp_time)]


    def _get_baseline_tp(self, subject, timepoints, time_list=None):
        if time_list is None:
            time_list = self._get_time_list(subject, timepoints)

        tp_id = list(time_list.keys())
        tp_time = list(time_list.values())
        return tp_id[np.argmin(tp_time)]

    def _undo_one_hot(self, y, dtype='float32'):
        y_true = np.zeros_like(y)
        for ul, it_ul in self.labels_lut.items():
            y_true[y == it_ul] = ul

        return y_true.astype(dtype)


    def process_subject(self, subject, force_flag=False, **kwargs):
        raise NotImplementedError

    def process_parallel(self, num_cores, **kwargs):
        self._on_pipeline_init()
        def _run(processing, subject, **kwargs):
            processing._update_subject_layout(subject)
            try:
                processing.process_subject(subject, **kwargs)
            except:
                return subject

        results = Parallel(n_jobs=num_cores, backend='threading')(delayed(_run)(
            copy.copy(self), subject, **kwargs) for subject in self.subject_list)

        print('Subjects that failed: ')
        print('\n'.join([r for r in results if r is not None]))
        return results

    def process(self, **kwargs):
        self._on_pipeline_init()

        subjects_failed = []
        for subject in self.subject_list:
            self._update_subject_layout(subject)
            try:
                self.process_subject(subject, **kwargs)
            except:
                subjects_failed += [subject]

        self._update_full_layout()
        print('Subjects that failed: ')
        print('\n'.join(subjects_failed))


class LabelFusionProcessing(Processing):
    def __init__(self, bids_loader, subject_list=None, pipeline_dir=None, **kwargs):
        self._pipeline_dir = pipeline_dir
        super(LabelFusionProcessing, self).__init__(bids_loader=bids_loader, subject_list=subject_list, **kwargs)

    def _name(self):
        return 'LabelFusion'

    @property
    def pipeline_dir(self):
        raise NotImplementedError("The pipeline_dir property should be implemented by sub-classes")

    def _get_tp_displacement(self, subject, target_tp, atlas_tp, **kwargs):
        raise NotImplementedError

    def _deform_atlases(self, subject, target_tp, timepoints, results_dir, *args, **kwargs):
        seg_ref_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
        if seg_ref_file is None:
            return

        proxysegref = nib.load(seg_ref_file.path)
        for atlas_tp in timepoints:
            im_file = self._get_data(**{**self.bf_entities, 'subject': subject, 'session': atlas_tp})
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': atlas_tp})

            if im_file is None or seg_file is None:
                continue

            output_image_filepath = join(results_dir, str(atlas_tp) + '.im.nii.gz')
            output_onehot_filepath = join(results_dir, str(atlas_tp) + '.onehot.nii.gz')

            # One-hot encoding of the labels
            proxyseg = nib.load(seg_file.path)
            seg_arr = np.array(proxyseg.dataobj)
            onehot_arr = one_hot_encoding(seg_arr, categories=self.labels_lut)

            # Gaussian filter for [1, 1, 1] segmentation
            proxyim = nib.load(im_file.path)
            arrayim = np.array(proxyim.dataobj)
            arrayim = gaussian_antialiasing(arrayim, proxyim.affine, [1, 1, 1])
            proxyim = nib.Nifti1Image(arrayim, proxyim.affine)
            proxyim = vol_resample_fast(proxyseg, proxyim)
            arrayim = np.array(proxyim.dataobj)

            if target_tp == atlas_tp:
                proxyonehot = nib.Nifti1Image(onehot_arr.astype('float32'), proxyseg.affine)
                nib.save(proxyonehot, output_onehot_filepath)
                nib.save(proxyim, output_image_filepath)

            else:
                tp_displ = self._get_tp_displacement(subject, target_tp, atlas_tp, **kwargs)

                # Image
                arrayim_mov = warp(arrayim, tp_displ)
                proxyim = nib.Nifti1Image(arrayim_mov.astype('float32'), proxysegref.affine)
                nib.save(proxyim, output_image_filepath)

                arrayonehot_mov = warp(onehot_arr, tp_displ)
                proxyonehot = nib.Nifti1Image(arrayonehot_mov.astype('float32'), proxysegref.affine)
                nib.save(proxyonehot, output_onehot_filepath)

    def _label_fusion(self, target_tp, timepoints, results_dir, **kwargs):
        raise NotImplementedError

    def _save_vols(self, vols, subject, *args, **kwargs):
        filepath = join(DIR_PIPELINES[self.pipeline_dir], 'sub-' + subject, 'sub-' + subject + '_vols.csv')
        if exists(filepath):
            data_df = pd.read_csv(filepath)
        else:
            data_df = pd.DataFrame(columns=['session', 'computed_from'] + list(self.labels_lut.keys()))

        for tp, tp_dict in vols.items():
            for cf, cf_dict in tp_dict.items():
                df = pd.DataFrame({'name': ['session', 'computed_from'] + list(cf_dict.keys()),
                                   'value': [tp, cf] + list(cf_dict.values())})
                data_df = pd.concat([df, data_df], ignore_index=True)

        data_df.set_index('session', inplace=True, drop=False)
        data_df.to_csv(filepath, index=False)

    def _get_vols(self, y, res=1, labels=None):
        if labels is None:
            labels = np.unique(y)

        n_dims = len(y.shape)
        if isinstance(res, int):
            res = [res] * n_dims
        vol_vox = np.prod(res)

        vols = {}
        for l in labels:
            mask_l = y == l
            vols[int(l)] = np.round(np.sum(mask_l) * vol_vox, 2)

        return vols

    def _get_vols_post(self, post, res=1):
        n_labels = post.shape[-1]
        n_dims = len(post.shape[:-1])
        if isinstance(res, int):
            res = [res] * n_dims
        vol_vox = np.prod(res)

        post /= np.sum(post, axis=-1, keepdims=True)

        vols = {}
        for l in range(n_labels):
            mask_l = post[..., l]
            vols[l] = np.round(np.sum(mask_l) * vol_vox, 2)

        return vols

    def _undo_one_hot(self, y, dtype='float32'):
        y_true = np.zeros_like(y)
        for ul, it_ul in self.labels_lut.items():
            y_true[y == it_ul] = ul

        return y_true.astype(dtype)

    def process_subject(self, subject, force_flag=False, *args, **kwargs):
        print('\nSubject: ' + subject)
        timepoints = self._get_timepoints(subject=subject, uslr=True)

        kwargs['time_list'] = self._get_time_list(subject, timepoints)

        if 'output_pipeline' in kwargs.keys():
            output_pipeline = kwargs['output_pipeline']
        else:
            output_pipeline = self.pipeline_dir

        vols_filepath = join(DIR_PIPELINES[output_pipeline], 'sub-' + subject, 'sub-' + subject + '_vols.csv')
        if exists(vols_filepath) and not force_flag:
            return

        # Save image and label registration
        sbj_vols = {}
        print(' * Timepoint: ', end=' ', flush=True)
        for target_tp in timepoints:
            print(target_tp, end='', flush=True)

            # I/O data
            seg_file = self._get_data(**{**self.seg_entities, 'subject': subject, 'session': target_tp})
            if seg_file is None:
                continue

            tmp_dir = join(self.tmp_dir, subject, str(target_tp))
            create_dir(tmp_dir)

            filename = self.bids_loader.build_path({'subject': subject, 'session': target_tp, 'extension': 'nii.gz',
                                                    'suffix': 'T1wdseg', 'acquisition': '1'},
                                                   absolute_paths=False, path_patterns=BIDS_PATH_PATTERN,
                                                   strict=False, validate=False)
            create_dir(dirname(join(DIR_PIPELINES[output_pipeline], filename)))

            # Main processes
            self._deform_atlases(subject, target_tp, timepoints, tmp_dir, **kwargs)
            seg, posteriors, v2r = self._label_fusion(target_tp, timepoints, tmp_dir, **kwargs)

            # Store results
            seg_true = self._undo_one_hot(seg, dtype='int16')
            img = nib.Nifti1Image(seg_true, v2r)
            nib.save(img, join(DIR_PIPELINES[output_pipeline], filename))

            pixdim = np.sqrt(np.sum(v2r * v2r, axis=0))[:-1]
            vols = self._get_vols(seg, res=pixdim, labels=list(self.labels_lut.values()))
            vols_post = self._get_vols_post(posteriors, res=pixdim)
            sbj_vols[target_tp] = {'seg': vols, 'post': vols_post}

            self._save_vols(sbj_vols, vols_filepath, **kwargs)

            remove_dir(tmp_dir)
            if target_tp == timepoints[-1]: print('.')
            else: print(',', end='', flush=True)


        print('DONE\n')

class GenerativeLabelFusionProcessing(LabelFusionProcessing):
    def _label_fusion(self, target_tp, timepoints, results_dir, time_scale=None, g_std=None, **kwargs):
        image_targ_filepath = join(results_dir, str(target_tp) + '.im.nii.gz')
        proxyim_targ = nib.load(image_targ_filepath)
        arrayim_targ = np.array(proxyim_targ.dataobj)
        # arrayim_targ = gaussian_antialiasing(arrayim_targ, proxyim_targ.affine, [1, 1, 1])

        arrayonehot_targ = None
        for atlas_tp in timepoints:
            image_filepath = join(results_dir, str(atlas_tp) + '.im.nii.gz')
            onehot_filepath = join(results_dir, str(atlas_tp) + '.onehot.nii.gz')
            if not exists(image_filepath) or not exists(onehot_filepath):
                continue

            proxyim = nib.load(image_filepath)
            arrayim = np.array(proxyim.dataobj)

            if g_std == None:
                g_ker = 1
            else:
                g_ker = 1 / np.sqrt(2 * np.pi) / g_std * np.exp(-0.5 / (g_std ** 2) * (arrayim_targ - arrayim) ** 2)

            if time_scale == None:
                t_ker = 1
            else:
                t_targ = kwargs['time_list'][target_tp]
                t_atlas = kwargs['time_list'][atlas_tp]
                t_ker = time_scale * np.exp(-time_scale * (t_targ - t_atlas))

            pdata = t_ker * g_ker
            if g_std != None or time_scale != None:
                pdata = pdata[..., np.newaxis]

            proxyonehot = nib.load(onehot_filepath)
            arrayonehot = np.array(proxyonehot.dataobj)

            if arrayonehot_targ is None:
                arrayonehot_targ = np.zeros(arrayonehot.shape)

            arrayonehot_targ += pdata * arrayonehot

        mask = np.sum(arrayonehot_targ[..., 1:], axis=-1) > 0
        arrayonehot_targ[~mask, 0] = 1
        arrayonehot_targ /= np.sum(arrayonehot_targ, axis=-1, keepdims=True)
        return np.argmax(arrayonehot_targ, axis=-1), arrayonehot_targ, proxyim_targ.affine

    def _save_vols(self, vols, filepath, time_scale=None, g_std=None, **kwargs):
        if exists(filepath):
            data_df = pd.read_csv(filepath, dtype=str)
        else:
            data_df = pd.DataFrame(columns=['session', 'method', 'time_scale', 'g_std'] + list(self.labels_dict.values()))

        data_df.set_index(['session', 'method'], drop=False, inplace=True)
        for tp, tp_dict in vols.items():
            for method, method_dict in tp_dict.items():
                if (tp, method) in data_df.index:
                    data_df.drop((tp, method), inplace=True)

                vols = {v: method_dict[self.labels_lut[k]] if self.labels_lut[k] in method_dict.keys() else 0 for k, v in self.labels_dict.items()}
                df = pd.Series({'session': tp, 'method': method, 'time_scale': time_scale, 'g_std': g_std, **vols})
                data_df = pd.concat([data_df, df.to_frame().T], ignore_index=True)
                data_df.set_index(['session', 'method'], drop=False, inplace=True)

        data_df.set_index('session', inplace=True, drop=False)
        data_df.to_csv(filepath, index=False)