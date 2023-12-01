#!/bin/bash

#SBATCH --output=../../job_logs/%j.out
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
source /scratch_net/biwidl319/jbermeo/conda/etc/profile.d/conda.sh
conda activate UIASegmentation
python -u ../main_training.py
