#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train_small
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate fairseq

set -x

bash /home/ms3682/code/musicbert_fork/test.sh \
    -d "$DATASETS_DIR"/octmidi_data_bin \
    -r /home/ms3682/code/musicbert_fork/musicbert \
    -a tiny \
    -w "25" \
    -u "150"

set +x
