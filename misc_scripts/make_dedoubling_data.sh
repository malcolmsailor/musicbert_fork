#!/usr/bin/env bash

set -e
VENV=write_chord_tones_seqs

module load miniconda
echo conda activate "${VENV}"
conda activate "${VENV}"

DEDOUBLE_FOLDER=~/code/dedoubling_data
cd $DEDOUBLE_FOLDER

WRITE_SEQS_FOLDER="/home/ms3682/code/write_seqs"
MUSICBERT_FOLDER="/home/ms3682/code/musicbert_fork"

TEMP_DIR=$(mktemp -d)
TEMP_DIR=$(readlink -f "${TEMP_DIR}")
DATASET_DIR=~/project/raw_data


set -x
DATASET_DIR=$DATASET_DIR python -m dedoubling_data \
    --config-file "$DEDOUBLE_FOLDER"/configs/csv.yaml \
    output_folder="${TEMP_DIR}"
bash ~/code/musicbert_fork/misc_scripts/write_seqs.sh \
    "$WRITE_SEQS_FOLDER"/configs/oct_data_dedoubling.yaml \
    "${TEMP_DIR}" \
    "/home/ms3682/project/datasets/dedoubling" \
    16 -o
set +x

rm -rf "${TEMP_DIR}"
