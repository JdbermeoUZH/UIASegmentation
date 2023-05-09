#!/bin/bash

#SBATCH --output=/scratch_net/biwidl210/kvergopoulos/SemesterProject/sbatch_logs/%j.out
#SBATCH --cpus-per-task=22
#SBATCH --mem=240G
source /scratch_net/biwidl210/kvergopoulos/conda/etc/profile.d/conda.sh
conda activate py11
python -u /scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/preprocessing/preprocess_script.py
