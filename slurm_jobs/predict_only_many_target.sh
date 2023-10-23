#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=predict_cts
#SBATCH --time=012:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

if [[ -z "$2" ]]; then
    echo Usage: predict_only_many_target.sh [run_name] [data_dir]
    exit 1
fi

RUN_NAME="$1"
DATA_DIR="$2"

module load miniconda
conda activate newbert

set -x


python training_scripts/train_chord_tones.py \
    -d "${DATA_DIR}" \
    --multitarget --skip-training --run-name "${RUN_NAME}"

set +x
