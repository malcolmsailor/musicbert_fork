#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=predict_quantize
#SBATCH --time=012:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

PREDICTIONS_BASE="/home/ms3682/project/saved_predictions/musicbert_unlabeled"
CHECKPOINT_BASE="/home/ms3682/project/new_checkpoints/musicbert_fork"

if [[ -z "$1" ]]; then
    echo Usage: save_predictions.sh [run_name]
    echo "   We look for a checkpoint in"
    echo "       $CHECKPOINT_BASE/[run_name]/checkpoint_best.pt"
    echo "   We save predictions to "
    echo "       $PREDICTIONS_BASE/[run_name]/"
    exit 1
fi

run_name="$1"
shift

module load miniconda
conda activate newbert

set -x

python /home/ms3682/code/musicbert_fork/eval_scripts/save_predictions.py \
    --data-dir /home/ms3682/project/datasets/ycac_no_salami_slice_bin/ \
    --checkpoint "${CHECKPOINT_BASE}"/"${run_name}"/checkpoint_best.pt \
    --output-folder "${PREDICTIONS_BASE}/${run_name}/" \
    "${@}"

set +x
