#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=C7gpu
#SBATCH --mail-user=lhalice@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/lhalice/EIS_fit_ECM_with_ML/out-c7gpu.log

python Regression_C7.py 