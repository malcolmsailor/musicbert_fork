#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=train_class
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate newbert

set -x

bash train_composer_classification.sh \
    -d /home/ms3682/project/datasets/composer_classification_data_bin \
    -r /home/ms3682/code/musicbert_fork/musicbert \
    -a base \
    -W composer_classification \
    -c /home/ms3682/project/checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt \
    -u 50000 \
    -w 10000

set +x



