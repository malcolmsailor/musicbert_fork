#!/bin/bash

#SBATCH --job-name=binarize_pretrain
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate fairseq

bash /home/ms3682/code/musicbert_fork/binarize_pretrain.sh /home/ms3682/project/octmidi
