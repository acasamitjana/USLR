import pdb
from os.path import exists, join, dirname
from argparse import ArgumentParser

import bids

from src.uslr import USLR_LabelFusion, USLR_Lin_LabelFusion
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
                    default='lin',
                    type=str,
                    choices=['l2', 'l1', 'bch-l2', 'bch-l2', 'lin', 'lin-reverse'],
                    help="cost of the graph optimization.")

parser.add_argument('--num_cores',
                    default=1,
                    type=int,
                    help="number of parallel processes to run the script.")

parser.add_argument('--t_scale',
                    default=None,
                    type=float,
                    help="scale of the exponential variable that models time kernel decay in label fusion.")

parser.add_argument('--g_std',
                    default=3,
                    type=float,
                    help="std of the gaussian variable that models image similarity in label fusion.")

args = parser.parse_args()

print('LOADING Dataset ...', end=' ', flush=True)

db_file = join(dirname(args.bids), 'BIDS-raw.db')
if not exists(db_file):
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False)
    bids_loader.save(db_file)
else:
    bids_loader = bids.layout.BIDSLayout(root=args.bids, validate=False, database_path=db_file)

bids_loader.add_derivatives(DIR_PIPELINES['preproc'])


if args.subjects is None:
    print('Total subjects found: N=' + str(len(bids_loader.get_subjects())), end='\n\n')
else:
    print('Total subjects found: N=' + str(len(args.subjects)), end='\n\n')

if args.cost in ['lin', 'lin-reverse']:
    bids_loader.add_derivatives(DIR_PIPELINES[args.cost])
    processing = USLR_Lin_LabelFusion(bids_loader=bids_loader, subject_list=args.subjects, pipeline_dir=args.cost)
    processing.process(force_flag=args.force, g_std=args.g_std, time_scale=args.t_scale, cost=args.cost)

elif args.cost == 'lin-template':
    pass

else:
    bids_loader.add_derivatives(DIR_PIPELINES['nonlin-' + args.cost])
    processing = USLR_LabelFusion(bids_loader=bids_loader, subject_list=args.subjects, pipeline_dir='nonlin-' + args.cost)
    processing.process(force_flag=args.force, g_std=args.g_std, time_scale=args.t_scale, cost=args.cost)
