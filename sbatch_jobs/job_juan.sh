#!/bin/bash
#SBATCH --output=job_logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

source /scratch_net/biwidl319/jbermeo/conda/etc/profile.d/conda.sh
conda activate UIASegmentation
cd /scratch_net/biwidl319/jbermeo/UIASegmentation/

python -u main_training.py "$@"
