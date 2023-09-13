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

bash /home/ms3682/code/musicbert_fork/training_scripts/train_chord_tones.sh \
    -d /home/ms3682/project/datasets/chord_tones/fairseq/chord_tones_seqs_bin \
    -a base \
    -W chord_tones_musicbert \
    -c "${SAVED_CHECKPOINTS_DIR}"/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt \
    -u 50000 \
    -w 10000

set +x

# Equivalent command for local development:
# bash train_chord_tones.sh \
#     -d /Users/malcolm/output/fairseq/chord_tones_seqs_bin \
#     -a base \
#     -W scratch \
#     -c /Volumes/Reicha/large_checkpoints/musicbert/checkpoint_last_musicbert_base.pt \
#     -u 50000 \
#     -w 10000

