import os
import json
import pdb

import tensorflow as tf
import neurite as ne

if 'NEURITE_BACKEND' not in os.environ:
    os.environ['NEURITE_BACKEND'] = 'tensorflow'

import subprocess

filename_entities = ['subject', 'session', 'run', 'acquisition', 'suffix', 'extension', 'task', 'tracer', 'reconstruction', 'desc', 'space']
BIDS_PATH_PATTERN = [
    # "sub-{subject}/sub-{subject}_space{space-<uslrlin|uslrnonlin>}[_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<aff|v2r>}{extension<.txt|.npy>|.npy}",
    # "sub-{subject}/sub-{subject}_space{space-<uslrlin|uslrnonlin>}[_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<T1w|T1wposteriors|svf|T1wmask|T1wdseg|space>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<aff|v2r>}{extension<.txt|.npy>|.npy}",
    "sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<T1w|T2w|T2star|T2starw|FLAIR|FLASH|PD|PDw|PDT2|inplaneT[12]|angio|dseg|posteriors|svf|jac|def|T1wstd|T1wmask|T1wdseg|T2wmask|T2wdseg|FLAIRmask|FLAIRdseg|mask|space>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<func>|func}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<bold|cbv|sbref>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
    "sub-{subject}[/ses-{session}]/{datatype<pet>|pet}/sub-{subject}[_ses-{session}][_space-{space}][_task-{task}][_acq-{acquisition}][_trc-{tracer}][_rec-{reconstruction}][_run-{run}][_part-{part}][_desc-{desc}]_{suffix<pet>}{extension<.nii|.nii.gz|.json|.txt|.npy>|.nii.gz}",
]

# MRI Templates
if 'PYTHONPATH' not in os.environ:
    print('Please, set up PYTHONPATH to the root of this project.')
    exit()

repo_home = os.environ.get('PYTHONPATH')
labels_registration = os.path.join(repo_home, 'data', 'labels_classes_priors', 'label_list_registration.npy')

MNI_TEMPLATE = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz')
MNI_SM_V2R = os.path.join(repo_home, 'data', 'atlas', 'mni_to_synthmorph_space.v2r.npy')

MNI_ATLAS_TEMPLATE = os.path.join(repo_home, 'data', 'atlas', 'mni_reg_to_synthmorph_atlas.nii.gz')
MNI_ATLAS_TEMPLATE_SEG = os.path.join(repo_home, 'data', 'atlas', 'mni_reg_to_synthmorph_atlas.seg.nii.gz')
MNI_ATLAS_TEMPLATE_MASK = os.path.join('data', 'atlas', 'mni_reg_to_synthmorph_atlas.mask.nii.gz')

MNI_TEMPLATE_SEG = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz')
MNI_TEMPLATE_MASK = os.path.join(repo_home, 'data', 'atlas', 'mni_icbm152_mask_tal_nlin_sym_09a.nii.gz')

DEBUG = False
if 'DEBUG' in os.environ:
    if os.environ['DEBUG'] == 'True' or os.environ['DEBUG'] == True or os.environ['DEBUG'] == '1': DEBUG = True

# BIDS directories ---- Environment variables.
BIDS_DIR = os.environ['BIDS_DIR']
if not BIDS_DIR: raise ValueError("Please, specify environment variable DB")

if BIDS_DIR[-1] == '/': BIDS_DIR = BIDS_DIR[:-1]
ROOT_DIR = os.path.dirname(BIDS_DIR)
USLR_DIR = os.path.join(ROOT_DIR, 'uslr')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TMP_DIR = os.path.join(ROOT_DIR, 'tmp')

if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
if not os.path.exists(TMP_DIR): os.makedirs(TMP_DIR)

DIR_PIPELINES = {
    'preproc': os.path.join(USLR_DIR, 'preproc'),
    'lin': os.path.join(USLR_DIR, 'lin'),
    'nonlin': os.path.join(USLR_DIR, 'nonlin'),
    'nonlin-l1': os.path.join(USLR_DIR, 'nonlin-l1'),
    'nonlin-l2': os.path.join(USLR_DIR, 'nonlin-l2'),
    'nonlin-bch-l1': os.path.join(USLR_DIR, 'nonlin-bch-l1'),
    'nonlin-bch-l2': os.path.join(USLR_DIR, 'nonlin-bch-l2'),
}

DESC_PIPELINES = {
    'preproc': 'USLR preprocessing',
    'lin': 'USRL linear registration stream',
    'nonlin': 'USRL non-linear registration stream',
    'nonlin-l1': 'USRL non-linear registration stream',
    'nonlin-l2': 'USRL non-linear registration stream',
    'nonlin-bch-l1': 'USRL non-linear registration stream',
    'nonlin-bch-l2': 'USRL non-linear registration stream',
}

for d, d_str in DESC_PIPELINES.items():
    if not os.path.exists(DIR_PIPELINES[d]): os.makedirs(DIR_PIPELINES[d])
    data_descr_path = os.path.join(DIR_PIPELINES[d], 'dataset_description.json')
    if not os.path.exists(data_descr_path):
        data_descr = {}
        data_descr['Name'] = os.path.basename(d_str)
        data_descr['BIDSVersion'] = '1.0.2'
        data_descr['GeneratedBy'] = [{'Name': d}]
        data_descr['Description'] = d_str
        data_descr_path = os.path.join(DIR_PIPELINES[d], 'dataset_description.json')
        json_object = json.dumps(data_descr, indent=4)
        with open(data_descr_path, 'w') as outfile:
            outfile.write(json_object)

# VERBOSE = os.environ['VERBOSE'] if 'VERBOSE' in os.environ.keys() else False
# if VERBOSE:
if 'USLR_RUNNING' not in os.environ:
    os.system('cls' if os.name == 'nt' else 'clear')
    print('          o')
    print('        ooooo')
    print('      ooooooooo')
    print('    ooooooooooooo')
    print('  ooooooooooooooooo')
    print('ooooooooooooooooooooo')
    print('')
    print('Running USLR Pipeline')
    print('')
    print('ooooooooooooooooooooo')
    print('ooooooooooooooooooooo')
    print('')

    if 'FREESURFER_SYNTHMORPH_HOME' in os.environ:
        subprocess.call(["bash", '-c', 'export FREESURFER_HOME=$FREESURFER_SYNTHMORPH_HOME'])
        subprocess.call(["bash", '-c', 'source $FREESURFER_HOME/SetUpFreeSurfer.sh'])
        print('- Freesurfer version for seg/reg is ' + os.environ['FREESURFER_SYNTHMORPH_HOME'])

    elif 'FREESURFER_HOME' in os.environ:
        print('- Freesurfer version for seg/reg is ' + os.environ['FREESURFER_HOME'])

    else:
        print('Please, source FREESURFER first for registration and segmentation.')
        exit()

    os.environ['USLR_RUNNING'] = 'True'

    print('- DATASET USED ($BIDS_DIR): ' + BIDS_DIR)
    print('- USLR DIR: ' + USLR_DIR)
    print('- RESULTS DIR: ' + RESULTS_DIR)
    print('')
    print('ooooooooooooooooooooo')
    print('ooooooooooooooooooooo')
    print('\n')




# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# device = 'cpu'

