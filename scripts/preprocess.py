import pdb
from os.path import exists, join, dirname
from argparse import ArgumentParser

import bids

from setup import *


parser = ArgumentParser(description='Corrects intensity inhomogeneities from T1w images and computes brain mask.')
parser.add_argument('--bids',
                    default=BIDS_DIR, help="specify the bids root directory (/rawdata)")
parser.add_argument('--subjects',
                    default=None, nargs='+', help="(optional) specify which subjects to process")
parser.add_argument('--num_cores',
                    default=1, type=int, help="number of parallel cores to run.")
parser.add_argument("--force", action='store_true',
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

if args.subjects is None:
    print('Total subjects found: N=' + str(len(bids_loader.get_subjects())), end=' ', flush=True)
else:
    print('Total subjects found: N=' + str(len(args.subjects)), end=' ', flush=True)


from src.uslr import USLRSegment, USLRBiasCorrection

# processing_seg = USLRSegment(bids_loader=bids_loader, subject_list=args.subjects)
# processing_seg.process(force_flag=False)

processing_bias = USLRBiasCorrection(bids_loader=bids_loader, subject_list=args.subjects)
if args.num_cores > 1:
    processing_bias.process_parallel(num_cores=args.num_cores, force_flag=args.force)
else:
    processing_bias.process(force_flag=args.force)

