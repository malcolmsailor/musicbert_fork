#!/bin/bash
#

# DATA_BIN_DIR=${1}_data_bin


TOTAL_UPDATES=125000
WARMUP_UPDATES=25000
PEAK_LR=0.0005
TOKENS_PER_SAMPLE=8192

# NB BATCH_SIZE is only used in the UPDATE_FREQ calculation below; the actual batch size
#   to fairseq-train is set by MAX_SENTENCES arg
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



while getopts "d:r:a:u:w:W:" opt; do
    case $opt in
        d) DATA_BIN_DIR="$OPTARG" ;;
        r) USER_DIR="$OPTARG" ;;
        a) NN_ARCH=musicbert_"$OPTARG" ;;
        u) TOTAL_UPDATES="$OPTARG" ;;
        w) WARMUP_UPDATES="$OPTARG" ;;
        W) WANDB_PROJECT="$OPTARG" ;;
        \?) echo "Usage: $(basename "$0") \
            -d data_dir \
            -r user_dir \
            -W wandb project \
            [-a architecture] \
            [-u total_updates] \
            [-w warmup steps]"
    esac
done

CHECKPOINT_SUFFIX=${NN_ARCH}
if [ -z "$DATA_BIN_DIR" ] || [ -z "$USER_DIR" ] || [ -z "$WANDB_PROJECT" ] || [ -z "$NN_ARCH" ]
then
    echo "-d data_dir, -r user dir, -a architecture, and -W wandb project are required"
    exit 1
fi

shift "$((OPTIND - 1))"

set -x
fairseq-train ${DATA_BIN_DIR} \
    --user-dir ${USER_DIR} \
    --wandb-project ${WANDB_PROJECT} \
    --restore-file checkpoints/checkpoint_last_${CHECKPOINT_SUFFIX}.pt \
    --task masked_lm --criterion masked_lm \
    --arch ${NN_ARCH} --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} \
    --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 1 \
    --checkpoint-suffix _${CHECKPOINT_SUFFIX}
set +x
