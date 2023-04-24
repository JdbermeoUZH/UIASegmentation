#!/bin/bash

#SBATCH --output=/scratch_net/biwidl210/kvergopoulos/SemesterProject/sbatch_logs/%j.out
#SBATCH --cpus-per-task=32
source /scratch_net/biwidl210/kvergopoulos/conda/etc/profile.d/conda.sh
conda activate py11_2
python -u /scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/preprocessing/preprocess_script.py
