import pdb
from os.path import exists, join, dirname
from argparse import ArgumentParser
from joblib import delayed, Parallel

import bids

from src.uslr import USLR_Linear
from setup import *

from os import listdir

parser = ArgumentParser(description='Runs USLR-rig step.')
parser.add_argument('--bids',
                    default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
parser.add_argument('--subjects',
                    default=None, nargs='+', help="(optional) specify which subjects to process")
parser.add_argument("--num_cores", default=1, type=int,
                    help="Number of cores in parallel processing")
parser.add_argument("--force", action='store_true',
                    help="Force the script to overwriting existing segmentations in the uslr/preproc directory.")
parser.add_argument("--reg_MNI", action='store_true',
                    help="Force the script to overwriting existing segmentations in the uslr/preproc directory.")


print('LOADING Dataset ...', end=' ', flush=True)
args = parser.parse_args()

db_file = join(dirname(args.bids), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)

bids_loader.add_derivatives(DIR_PIPELINES['preproc'])
bids_loader.add_derivatives(DIR_PIPELINES['lin'])

processing = USLR_Linear(bids_loader=bids_loader, subject_list=args.subjects)

if args.subjects is None:
    print('Total subjects found: N=' + str(len(processing.get_subjects(uslr=True))), end='\n\n')
else:
    print('Total subjects found: N=' + str(len(args.subjects)), end='\n\n')

print('\n')

if args.num_cores > 1:
    processing.process_parallel(num_cores=args.num_cores, force_flag=args.force, register_MNI=args.reg_MNI)
else:
    processing.process(force_flag=args.force, register_MNI=args.reg_MNI)
