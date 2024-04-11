#!/bin/bash

#SBATCH --job-name=write_seqs
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

INPUT_DIR=$1
INTERMEDIATE_DIR=$2 # E.g., ~/project/raw_data/abstract_seqs
RAW_OUTPUT_DIR=$3   # E.g., ~/project/datasets/chord_tones/fairseq/abstract_raw
INPUTS_PATHS=$4
if [[ -z "$4" ]]; then
    echo Error: 2 positional arguments required.
    echo Usage: bash write_abstract_seqs.sh [input_dir] [intermediate_dir] [output_dir] [input_splits_paths]
    exit 1
fi

set -e

if [[ -d "${RAW_OUTPUT_DIR/_raw/_bin}" ]]; then
    echo Error: "${RAW_OUTPUT_DIR/_raw/_bin}" exists
fi

module load miniconda
conda activate write_chord_tones_seqs

set -x
python -m write_seqs \
    --src-data-dir ${INPUT_DIR} \
    --data-settings ~/code/write_seqs/configs/oct_data_abstract.yaml \
    --output-dir ${INTERMEDIATE_DIR} \
    --input-paths-dir ${INPUTS_PATHS}

python ~/code/write_seqs/scripts/to_fair_seq_abstract.py \
    --input-dir ${INTERMEDIATE_DIR} \
    --output-dir ${RAW_OUTPUT_DIR}
set +x

conda activate newbert

set -x
python ~/code/musicbert_fork/binarize_scripts/binarize_abstract_folder.py \
    input_folder=${RAW_OUTPUT_DIR} workers=16

set +x
