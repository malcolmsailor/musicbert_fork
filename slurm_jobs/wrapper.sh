#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=gpu-wrapper
#SBATCH --time=012:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate newbert

set -x

"${@}"

set +x
