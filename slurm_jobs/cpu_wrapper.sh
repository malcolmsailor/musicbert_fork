#!/bin/bash

#SBATCH --job-name=write_seqs
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate newbert

set -x

"${@}"

set +x
