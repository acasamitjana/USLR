# USLR: an open-source tool for unbiased and smooth longitudinal registration of brain MRI

This repository performs linear and nonlinear registration between a set of points (e.g., timepoints in longitudinal studies) and a shared latent space. We use the log-space of transforms to infere the most probable deformations using Bayesian inference


### Requirements:
**Python** <br />
The code run on python v3.10 and several external libraries listed under _requirements.txt_

**BIDS protocol** <br />
The pipeline works with datasets following the BIDS protocol. 

**Freesurfer installed**<br />
We use Synthseg and Synthmorph, two learning-based functionalities of freesurfer for MRI processing. We use freesurfer version 3.8. Please, make sure that freesurfer is properly sourced.

**GPU (optional)**<br />
If a GPU is available, non-linear stream of the pipeline will run faster.

**Data**<br />
Data needs to be organised following the BIDS protocol. Important! Make sure that
if multiple T1w images are available, the difference is not in the _acquisition_
entity (it can be in other, most often _run_, but also _desc_, _space_, etc. ). 
### Run the code
- **Set-up environmental variables** 
  - BIDS_DIR: your path to the 'rawdata' directory of the BIDS protocol
  - PYTHONPATH: your path to root directory of this project.
- **Run pre-processing**
   - _scripts/preprocess.sh_: this script will run over all subjects available in $BIDS_DIR. It also accepts a list of arguments (SUBJECT_ID) to run it over a subset (1, ..., N) subjects. It performs anatomical segmentation using and intensity inhomogeneity correction. The output will be stored in $BIDS_DIR/../uslr/preproc.
- **Run linear registration**
  - _scripts/linear_registration.py_: this script will run over all subjects available in $BIDS_DIR/../uslr/preproc. It also accepts a list of arguments (SUBJECT_ID) to run it over a subset of (1, ..., N) subjects. Please check the other available options. The output will be stored in $BIDS_DIR/../uslr/lin
- **Run non-linear registration**
  - _scripts/run_nonlinear_registration.sh_: this script will run over all subjects available in $BIDS_DIR/../uslr/lin (subjects processed using the linear registration script). It also accepts a list of arguments (SUBJECT_ID) to run it over a subset (1, ..., N) subjects. Please check the other available options. The output will be stored in $BIDS_DIR/../uslr/nonlin-bch-l2 by default (or similaraly depending on the cost function used).


## Code updates

18 February 2024:
- Version 2 of USLR. It uses an updated version of SynthMorph available with Freesurfer3.8. It also simplifies the pipeline functionalities in order to be run, with clearer terminal messages.

10 November 2023:
- Initial commit and README file.



## Citation
TBC



