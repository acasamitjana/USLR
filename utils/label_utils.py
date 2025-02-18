import pdb
from os.path import join, dirname, exists

import csv
import numpy as np

from setup import *


repo_home = os.environ.get('PYTHONPATH')
ctx_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_labels.npy'))
ctx_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_parcellation_names.npy'))
APARC_DICT = {int(k): str(v) for k, v in zip(ctx_labels, ctx_names) if v.lower() != 'background'}
APARC_DICT_REV = {v: k for k, v in zip(ctx_labels, ctx_names) if v.lower() != 'background'}

subcortical_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_labels.npy'))
subcortical_labels = np.concatenate((subcortical_labels, [24]))
subcortical_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'synthseg_segmentation_names.npy'))
subcortical_names = np.concatenate((subcortical_names, ['csf']))
SYNTHSEG_DICT = {int(k): str(v) for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}
SYNTHSEG_DICT_REV = {v: k for k, v in zip(subcortical_labels, subcortical_names) if v.lower() != 'background'}
SYNTHSEG_APARC_DICT = {**SYNTHSEG_DICT, **APARC_DICT}
SYNTHSEG_APARC_DICT_REV = {**SYNTHSEG_DICT_REV, **APARC_DICT_REV}

aseg_names = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'aseg_segmentation_names.npy'))
aseg_labels = np.load(os.path.join(repo_home, 'data', 'labels_classes_priors', 'aseg_segmentation_labels.npy'))
ASEG_DICT = {k: v for k, v in zip(aseg_labels, aseg_names) if v.lower() != 'background'}
ASEG_DICT_REV = {v: k for k, v in zip(aseg_labels, aseg_names) if v.lower() != 'background'}

SYNTHSEG_LUT = {k: it_k for it_k, k in enumerate(np.unique(subcortical_labels))}
SYNTHSEG_LUT = {**SYNTHSEG_LUT, **{k: SYNTHSEG_LUT[3] if k < 2000 else SYNTHSEG_LUT[42] for k in ctx_labels if k != 0}}
ASEG_LUT = {k: it_k for it_k, k in enumerate(np.unique(aseg_labels))}
ASEG_LUT = {**ASEG_LUT, **{k: ASEG_LUT[3] if k < 2000 else ASEG_LUT[42] for k in ctx_labels if k != 0}}
SYNTHSEG_APARC_LUT = {k: it_k for it_k, k in enumerate(np.unique(np.concatenate((subcortical_labels, ctx_labels), axis=0)))}
ASEG_APARC_LUT = {k: it_k for it_k, k in enumerate(np.unique(np.concatenate((aseg_labels, ctx_labels), axis=0)))}


CLUSTER_DICT = {
    'Gray': [53, 17, 51, 12, 54, 18, 50, 11, 58, 26, 42, 3],
    'CSF': [4, 5, 43, 44, 15, 14, 24],
    'Thalaumus': [49, 10],
    'Pallidum': [52, 13],
    'VentralDC': [28, 60],
    'Brainstem': [16],
    'WM': [41, 2],
    'cllGM': [47, 8],
    'cllWM': [46, 7]
}

CSF_LABELS = [24] # CLUSTER_DICT['CSF']


if not exists(join(DERIVATIVES_DIR, 'synthseg_lut.txt')):

    labels_abbr = {
        0: 'BG',
        2: 'L-Cerebral-WM',
        3: 'L-Cerebral-GM',
        4: 'L-Lat-Vent',
        5: 'L-Inf-Lat-Vent',
        7: 'L-Cerebell-WM',
        8: 'L-Cerebell-GM',
        10: 'L-TH',
        11: 'L-CAU',
        12: 'L-PU',
        13: 'L-PA',
        14: '3-Vent',
        15: '4-Vent',
        16: 'BS',
        17: 'L-HIPP',
        18: 'L-AM',
        26: 'L-ACC',
        28: 'L-VDC',
        41: 'R-Cerebral-WM',
        42: 'R-Cerebral-GM',
        43: 'R-Lat-Vent',
        44: 'R-Inf-Lat-Vent',
        46: 'R-Cerebell-WM',
        47: 'R-Cerebell-WM',
        49: 'R-TH',
        50: 'R-CAU',
        51: 'R-PU',
        52: 'R-PA',
        53: 'R-HIPP',
        54: 'R-AM',
        58: 'R-ACC',
        60: 'R-VDC',
    }

    fs_lut = {0: {'name': 'Background', 'R': 0, 'G': 0, 'B': 0}}
    with open(join(os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt'), 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            info = [r for r in row[None][0].split(' ') if r != '']
            if len(info) < 5: continue
            try:
                name = info[1].lower().replace('-', ' ')
                fs_lut[int(info[0])] = {'name': name, 'R': info[2], 'G': info[3], 'B': info[4]}
            except:
                continue

    header = ['index', 'name', 'abbreviation', 'R', 'G', 'B', 'mapping']
    label_dict = [
        {'index': label, 'name': fs_lut[label]['name'],
         'abbreviation': labels_abbr[label] if label in labels_abbr else fs_lut[label]['name'],
         'R': fs_lut[label]['R'], 'G': fs_lut[label]['G'], 'B': fs_lut[label]['B'], 'mapping': it_label}
        for label, it_label in SYNTHSEG_APARC_LUT.items()
    ]

    with open(join(DERIVATIVES_DIR, 'synthseg_lut.txt'), 'w') as csvfile:
        csvreader = csv.DictWriter(csvfile, fieldnames=header, delimiter='\t')
        csvreader.writeheader()
        csvreader.writerows(label_dict)