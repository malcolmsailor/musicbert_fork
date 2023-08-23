#!/bin/bash
#

# DATA_BIN_DIR=${1}_data_bin


TOTAL_UPDATES=125000
WARMUP_UPDATES=25000
PEAK_LR=0.0005
TOKENS_PER_SAMPLE=8192
BATCH_SIZE=256
MAX_SENTENCES=4

if command -v nvidia-smi > /dev/null 2>&1 ;
then
    N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    N_GPU_LOCAL=0
fi

UPDATE_FREQ_DENOM=$((N_GPU_LOCAL>1 ? N_GPU_LOCAL : 1))
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${UPDATE_FREQ_DENOM}))

CHECKPOINT_SUFFIX=${NN_ARCH}

while getopts "d:r:a:u:w" opt; do
    case $opt in
        d) DATA_BIN_DIR="$OPTARG" ;;
        r) USER_DIR="$OPTARG" ;;
        a) NN_ARCH=musicbert_"$OPTARG" ;;
        u) TOTAL_UPDATES="$OPTARG" ;;
        w) WARMUP_UPDATES="$OPTARG" ;;
        \?) echo "Usage: $(basename "$0") \
            -d data_dir \
            -r user_dir \
            [-a architecture] \
            [-u total_updates] \
            [-w warmup steps]"
    esac
done

if [ -z "$DATA_BIN_DIR" ] || [ -z "$USER_DIR" ]
then
    echo "-d data_dir and -r user dir are required"
    exit 1
fi

echo WARMUP_UPDATES=$WARMUP_UPDATES
echo TOTAL_UPDATES=$TOTAL_UPDATES

exit 0


shift "$((OPTIND - 1))"

set -x
fairseq-train ${DATA_BIN_DIR} \
    --user-dir ${USER_DIR} \
    --restore-file checkpoints/checkpoint_last_${CHECKPOINT_SUFFIX}.pt \
    --task masked_lm --criterion masked_lm \
    --arch ${NN_ARCH} --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} \
    --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 100 \
    --checkpoint-suffix _${CHECKPOINT_SUFFIX}
set +x
