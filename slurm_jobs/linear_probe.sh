#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=lin_probe
#SBATCH --time=012:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate newbert

set -x

python ~/code/musicbert_fork/experiments/linear_probe/linear_probe.py \
    data_dir=~/project/datasets/labeled_bach_chorales_bin \
    checkpoint=~/project/new_checkpoints/musicbert_fork/32702693/checkpoint_best.pt \
    ref_dir=~/project/datasets/chord_tones/fairseq/many_target_bin \
    "${@}"

set +x
