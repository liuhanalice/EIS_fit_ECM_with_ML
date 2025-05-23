#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=C4
#SBATCH --mail-user=lhalice@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/home/lhalice/EIS_fit_ECM_with_ML/out-c4.log

python Regression_C4.py 