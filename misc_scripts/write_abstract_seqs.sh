#!/bin/bash

#SBATCH --job-name=write_seqs
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

INPUT_DIR=$1

set -e

module load miniconda
conda activate write_chord_tones_seqs

set -x
SRC_DATA_DIR=${INPUT_DIR} python -m write_seqs \
    --data-settings ~/code/write_seqs/configs/oct_data_abstract.yaml \
    --output-dir ~/project/raw_data/abstract_seqs --frac 0.03

python ~/code/write_seqs/scripts/to_fair_seq_abstract.py \
    --input-dir ~/project/raw_data/abstract_seqs \
    --output-dir ~/project/datasets/chord_tones/fairseq/abstract_raw
set +x

conda activate newbert

set -x
python ~/code/musicbert_fork/binarize_scripts/binarize_abstract_folder.py \
    input_folder=~/project/datasets/chord_tones/fairseq/abstract_raw

set +x
