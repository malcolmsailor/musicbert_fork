#!/bin/bash

#SBATCH --job-name=write_seqs
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

if [[ -z "$4" ]]; then
    echo Error: 4 positional arguments required.
    echo Usage: bash write_seqs.sh [data_settings] [src_data_dir_or_zip] [output_dir] [n_workers] [-o]
    exit 1
fi

if [[ -z "${WRITE_SEQS_ENV// /}" ]]; then
    WRITE_SEQS_ENV=~/venvs/write_seqs
fi

if [[ -z "${WRITE_SEQS_FOLDER// /}" ]]; then
    WRITE_SEQS_FOLDER=~/google_drive/python/data_science/write_seqs
fi

if [[ -z "${MUSICBERT_ENV// /}" ]]; then
    MUSICBERT_ENV=~/venvs/musicbert_fork
fi

if [[ -z "${MUSICBERT_FOLDER// /}" ]]; then
    MUSICBERT_FOLDER=~/google_drive/python/data_science/musicbert_fork
fi

DATA_SETTINGS=$(readlink -f "${1}")
INPUT_DIR=$(readlink -f "${2}")
TEMP_DIR=$(mktemp -d)
TEMP_DIR=$(readlink -f "${TEMP_DIR}")

if [ ! -d "${3}" ]; then
    mkdir -p "${3}"
fi

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

# echo conda activate "${WRITE_SEQS_ENV}"
# conda activate "${WRITE_SEQS_ENV}"
echo source "${WRITE_SEQS_ENV}"/bin/activate
source "${WRITE_SEQS_ENV}"/bin/activate

if [[ "${INPUT_DIR}" =~ .*\.zip$ ]]; then
    INPUT_DIR_TMP="${INPUT_DIR%.zip}"
    unzip -o "${INPUT_DIR}" -d "${INPUT_DIR_TMP}"
    INPUT_DIR="${INPUT_DIR_TMP}"
fi

# I'm not sure if the module is installed in the env so we cd into the directory to
#   be sure it will run
set -e
set -x
cd "${WRITE_SEQS_FOLDER}"
SRC_DATA_DIR="${INPUT_DIR}" python -m write_seqs \
    --data-settings "${DATA_SETTINGS}" --output-dir "${TEMP_DIR}" "${@:6}"

cd "${WRITE_SEQS_FOLDER}"
python scripts/to_fair_seq.py \
    --input-dir "${TEMP_DIR}" \
    --output-dir "${OUTPUT_DIR}"_raw
rm -rf "${TEMP_DIR}"
set +x

# echo conda activate "${MUSICBERT_ENV}"
# conda activate "${MUSICBERT_ENV}"
echo source "${MUSICBERT_ENV}"/bin/activate
source "${MUSICBERT_ENV}"/bin/activate

set -x
cd "${MUSICBERT_FOLDER}"
bash binarize_scripts/binarize_chord_tones.sh "${OUTPUT_DIR}"_raw "${N_WORKERS}"
set +x
