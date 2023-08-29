# TODO: (Malcolm 2023-08-29) restore after completing this script
# if [[ `git status --porcelain` ]]; then
#   echo "There are uncommitted changes; commit them then rerun"
#   exit 1
# fi

TOTAL_UPDATES=125000
WARMUP_UPDATES=25000

PEAK_LR=0.0005 # Borrowed from musicbert

# NB in musicbert scripts, BATCH_SIZE is only used in the UPDATE_FREQ calculation below;
#   the actual batch size to fairseq-train is set by MAX_SENTENCES arg
BATCH_SIZE=64
MAX_SENTENCES=4

MAX_POSITIONS=8192 # TODO: (Malcolm 2023-08-29) update

HEAD_NAME="composer_classification"
NUM_CLASSES=7  # TODO make a parameter or something


if command -v nvidia-smi > /dev/null 2>&1 ;
then
    N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    CPU_FLAG=""
else
    N_GPU_LOCAL=0
    CPU_FLAG=--cpu
fi

UPDATE_FREQ_DENOM=$((N_GPU_LOCAL>1 ? N_GPU_LOCAL : 1))
UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${UPDATE_FREQ_DENOM}))

while getopts "d:r:a:u:w:W:c:" opt; do
    case $opt in
        d) DATA_BIN_DIR="$OPTARG" ;;
        r) USER_DIR="$OPTARG" ;;
        a) NN_ARCH=musicbert_"$OPTARG" ;;
        u) TOTAL_UPDATES="$OPTARG" ;;
        w) WARMUP_UPDATES="$OPTARG" ;;
        W) WANDB_PROJECT="$OPTARG" ;;
        c) RESTORE_CHECKPOINT="$OPTARG" ;;
        \?) echo "Usage: $(basename "$0") \
            -d data_dir \
            -r user_dir \
            -W wandb project \
            -a architecture \
            [-u total_updates] \
            [-w warmup steps]"
    esac
done

if [ -z "$DATA_BIN_DIR" ] || [ -z "$USER_DIR" ] || [ -z "$WANDB_PROJECT" ] || [ -z "$NN_ARCH" ]
then
    echo "-d data_dir, -r user dir, -a architecture, and -W wandb project are required"
    exit 1
fi

if [ -z "$RESTORE_CHECKPOINT" ]
then
    RESTORE_FLAG=""
else
    RESTORE_FLAG="--restore-file ${RESTORE_CHECKPOINT}"
fi



FAIRSEQ_ARGS=(
    ${DATA_BIN_DIR}
    ${CPU_FLAG}
    --user-dir ${USER_DIR}
    ${RESTORE_FLAG}
    --wandb-project ${WANDB_PROJECT}
    --task sentence_prediction
    --arch ${NN_ARCH}
    --batch-size $MAX_SENTENCES 
    --update-freq $UPDATE_FREQ 

    --criterion sentence_prediction
    --classification-head-name ${HEAD_NAME}
    --num-classes ${NUM_CLASSES}

    # These `reset` params seem to be required for fine-tuning
    --reset-optimizer
    --reset-dataloader
    --reset-meters

    # Hyperparameters directly from musicbert scripts:
    --optimizer adam 
    --adam-betas '(0.9,0.98)'
    --adam-eps 1e-6 
    --clip-norm 0.0
    --lr-scheduler polynomial_decay 
    --lr ${PEAK_LR}
    --log-format simple
    --log-interval 1 # TODO restore a higher value
    --warmup-updates ${WARMUP_UPDATES} 
    --total-num-update ${TOTAL_UPDATES}
    --max-update ${TOTAL_UPDATES}
    --shorten-method 'truncate'
    --no-epoch-checkpoints
    --find-unused-parameters

    # TODO: (Malcolm 2023-08-29) update best checkpoint metric
    --best-checkpoint-metric accuracy 
    --maximize-best-checkpoint-metric

    # Args from musicbert not (yet?) using:
    --max-positions 8192

    --required-batch-size-multiple 1
    --init-token 0 --separator-token 2
    # TODO: (Malcolm 2023-08-29) not sure what --max-tokens does
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES}))

    # Why does musicbert set num workesr to 0???
    # TODO: (Malcolm 2023-08-29) set number of workers
    # --num-workers 0 \
)

set -x

fairseq-train "${FAIRSEQ_ARGS[@]}"

set +x


