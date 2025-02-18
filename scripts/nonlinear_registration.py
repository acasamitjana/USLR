import pdb
from os.path import exists, join, dirname, isdir
from argparse import ArgumentParser

import bids

from src.uslr import USLR_Deformable
from setup import *


parser = ArgumentParser(description='Corrects intensity inhomogeneities from T1w images and computes brain mask.')
parser.add_argument('--bids',
                    default=BIDS_DIR,
                    help="specify the bids root directory (/rawdata)")

parser.add_argument('--subjects',
                    default=None,
                    nargs='+',
                    help="(optional) specify which subjects to process")

parser.add_argument("--force",
                    action='store_true',
                    help="Force the script to overwriting existing segmentations in the derivatives/synthseg directory.")

parser.add_argument('--cost',
                    default='bch-l2',
                    type=str,
                    choices=['l2', 'l1', 'bch-l2', 'bch-l2'],
                    help="cost of the graph optimization.")

parser.add_argument('--num_cores',
                    default=1,
                    type=int,
                    help="number of parallel processes to run the script.")

parser.add_argument("--not_keep_wrong",
                    action='store_false',
                    help="Keep wrong segmentation files if bias field could not be computed")

args = parser.parse_args()

print('LOADING Dataset ...', end=' ', flush=True)

db_file = join(dirname(args.bids), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)

bids_loader.add_derivatives(DIR_PIPELINES['preproc'])
bids_loader.add_derivatives(DIR_PIPELINES['lin'])
bids_loader.add_derivatives(DIR_PIPELINES['nonlin-' + args.cost])

if args.subjects is None:
    print('Total subjects found: N=' + str(len(bids_loader.get_subjects())), end='\n\n')
else:
    print('Total subjects found: N=' + str(len(args.subjects)), end='\n\n')

processing = USLR_Deformable(bids_loader=bids_loader, subject_list=args.subjects)
processing.process(force_flag=args.force, cost=args.cost)
