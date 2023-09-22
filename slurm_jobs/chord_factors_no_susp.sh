#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=train_cts
#SBATCH --time=012:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate newbert

set -x

python /home/ms3682/code/musicbert_fork/training_scripts/train_chord_tones.py \
    -d /home/ms3682/project/datasets/chord_tones/fairseq/chord_factors_bin \
    -a base \
    -W chord_factors_musicbert \
    "${@}"

set +x


