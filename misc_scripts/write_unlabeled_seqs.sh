#!/bin/bash

#SBATCH --job-name=write_unlabeled_seqs
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

if [[ -z "$4" ]]; then
    echo Error: 4 positional arguments required.
    echo Usage: bash write_seqs.sh [src_data_dir_or_zip] [ref_dir] [output_dir] [n_workers] [-o]
    exit 1
fi

if [[ -z "${WRITE_SEQS_ENV// /}" ]]; then
    WRITE_SEQS_ENV=write_chord_tones_seqs
fi

if [[ -z "${WRITE_SEQS_FOLDER// /}" ]]; then
    WRITE_SEQS_FOLDER="/home/ms3682/code/write_seqs"
fi

if [[ -z "${MUSICBERT_ENV// /}" ]]; then
    MUSICBERT_ENV=newbert
fi

if [[ -z "${MUSICBERT_FOLDER// /}" ]]; then
    MUSICBERT_FOLDER="/home/ms3682/code/musicbert_fork"
fi

module load miniconda

INPUT_DIR=$(readlink -f "${1}")
REF_DIR=$(readlink -f "${2}")
# We need to add a subdirectory to the temporary directory because the
#   write_unlabeled_seqs.py script expects the destination not to exist
TEMP_DIR=$(mktemp -d)/temporary_data
TEMP_DIR=$(readlink -f "${TEMP_DIR}")
OUTPUT_DIR=$(readlink -f "${3}")
N_WORKERS="${4}"
[[ "$5" == "-o" ]] && OVERWRITE=true || OVERWRITE=false

if [[ -d "${OUTPUT_DIR}_raw" ]]; then
    if $OVERWRITE; then
        rm -rf "${OUTPUT_DIR}_raw"
    else
        echo Error, output_dir "${OUTPUT_DIR}_raw" exists
        exit 1
    fi
fi

if [[ -d "${OUTPUT_DIR}_bin" ]] && $OVERWRITE; then
    rm -rf "${OUTPUT_DIR}_bin"
fi

echo conda activate "${WRITE_SEQS_ENV}"
conda activate "${WRITE_SEQS_ENV}"

if [[ "${INPUT_DIR}" =~ .*\.zip$ ]]; then
    INPUT_DIR_TMP="${INPUT_DIR%.zip}"
    unzip -o "${INPUT_DIR}" -d "${INPUT_DIR_TMP}"
    INPUT_DIR="${INPUT_DIR_TMP}"
fi

set -e
set -x
python "${WRITE_SEQS_FOLDER}"/scripts/write_unlabeled_seqs.py \
    input_folder="${INPUT_DIR}" output_folder="${TEMP_DIR}" num_workers="${N_WORKERS}"

python "${WRITE_SEQS_FOLDER}"/scripts/to_fair_seq.py \
    --input-dir "${TEMP_DIR}" \
    --output-dir "${OUTPUT_DIR}"_raw
rm -rf "${TEMP_DIR}"
set +x

# If I recall correctly I do `set +x` then `echo` then `set -x` again because conda
#   activate itself does a lot of commands which with `set -x` will be printed
echo conda activate "${MUSICBERT_ENV}"
conda activate "${MUSICBERT_ENV}"

set -x
bash "${MUSICBERT_FOLDER}"/binarize_scripts/binarize_chord_tones_unlabeled.sh \
    "${OUTPUT_DIR}"_raw "${REF_DIR}" "${N_WORKERS}"
set +x
