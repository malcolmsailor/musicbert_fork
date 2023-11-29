if [ "$#" -ne 1 ]; then
    echo "Usage: $0 TASK"
    exit 1
fi

TASK=$1

set -x
sbatch ~/code/musicbert_fork/misc_scripts/write_seqs.sh \
    ~/code/write_seqs/configs/oct_data_"${TASK}".yaml \
    ~/project/raw_data/salami_slice_no_suspensions \
    ~/project/datasets/chord_tones/fairseq/"${TASK}" \
    16 -o
set +x
