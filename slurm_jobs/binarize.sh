#!/bin/bash

#SBATCH --job-name=binarize
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work

module load miniconda
conda activate newbert

if [[ -z "$1" ]]
then
    echo Error: requires at least one positional argument.
    echo    Usage: bash write_seqs.sh \[input data ending in _raw] ...
fi

set -x
bash /home/ms3682/code/musicbert_fork/binarize_scripts/binarize_chord_tones.sh "${@}"
set +x
