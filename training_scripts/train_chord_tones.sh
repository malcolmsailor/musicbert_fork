# For the sake of reproducibility, we want to commit all changes in the directory before
# running. Then we can get the git hash and command from wandb to reproduce (hopefully!)

if [[ -z "DEBUG_MUSICBERT" ]] && [[ `git status --porcelain` ]]; then
  echo "There are uncommitted changes; commit them then rerun"
  exit 1
fi

TOTAL_UPDATES=125000
WARMUP_UPDATES=25000

# TODO: (Malcolm 2023-09-11) restore
# PEAK_LR=0.0005 # Borrowed from musicbert
PEAK_LR=0.01

# NB in musicbert scripts, BATCH_SIZE is only used in the UPDATE_FREQ calculation below;
#   the actual batch size to fairseq-train is set by MAX_SENTENCES arg
BATCH_SIZE=64
MAX_SENTENCES=4

TOKENS_PER_SAMPLE=8192

HEAD_NAME="sequence_tagging_head"



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
LOG_INTERVAL=50

SCRIPT_DIR=$(dirname "$0")
USER_DIR=${SCRIPT_DIR}/../musicbert

DEBUG_FLAG=""

while getopts "d:u:a:w:W:c:l:Dh" opt; do
    case $opt in
        d) DATA_BIN_DIR="$OPTARG" ;;
        a) NN_ARCH=musicbert_"$OPTARG" ;;
        u) TOTAL_UPDATES="$OPTARG" ;;
        w) WARMUP_UPDATES="$OPTARG" ;;
        W) WANDB_PROJECT="$OPTARG" ;;
        c) RESTORE_CHECKPOINT="$OPTARG" ;;
        l) LOG_INTERVAL="$OPTARG" ;;
        D) DEBUG_FLAG="--msdebug" ;;
        h) echo "Usage: $(basename "$0") \
            -d data_dir \
            -W wandb project \
            -a architecture \
            [-u total_updates] \
            [-w warmup steps] \
            [-D] enable debug breakpoint hook"
    esac
done


if [ -z "$DATA_BIN_DIR" ] || [ -z "$WANDB_PROJECT" ] || [ -z "$NN_ARCH" ]
then
    echo "-d data_dir, -a architecture, and -W wandb project are required"
    exit 1
fi

if [ -z "$RESTORE_CHECKPOINT" ]
then
    RESTORE_FLAG=""
else
    RESTORE_FLAG="--restore-file ${RESTORE_CHECKPOINT}"
fi

NUM_CLASSES=$(cat "${DATA_BIN_DIR}"/label/dict.txt | grep -v -E "madeupword[0-9]{4}" | wc -l)

FAIRSEQ_ARGS=(
    ${DATA_BIN_DIR}
    ${CPU_FLAG}
    --user-dir ${USER_DIR}
    ${RESTORE_FLAG}
    # TODO: (Malcolm 2023-09-12) remove this flag
    --freeze-layers 1
    ${DEBUG_FLAG}
    --wandb-project ${WANDB_PROJECT}
    --task musicbert_sequence_tagging
    --arch ${NN_ARCH}
    --batch-size $MAX_SENTENCES 
    --update-freq $UPDATE_FREQ 

    --criterion sequence_tagging
    --classification-head-name ${HEAD_NAME}
    --compound-token-ratio 8
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
    --log-interval ${LOG_INTERVAL}
    --warmup-updates ${WARMUP_UPDATES} 
    --total-num-update ${TOTAL_UPDATES}
    --max-update ${TOTAL_UPDATES}
    --no-epoch-checkpoints
    --find-unused-parameters

    # TODO: (Malcolm 2023-08-29) update best checkpoint metric (f1?)
    --best-checkpoint-metric accuracy 
    --maximize-best-checkpoint-metric

    # I believe we need to keep max positions the same as musicbert
    --max-positions 8192

    --required-batch-size-multiple 1

    # --shorten-method, --init-token, and --separator-token are unrecognized
    #   arguments for token classification. TODO I should inspect this further.
    # --shorten-method 'truncate'
    # --init-token 0 --separator-token 2

    # TODO: (Malcolm 2023-08-29) not sure what --max-tokens does
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES}))

    # Why does musicbert set num workesr to 0???
    # TODO: (Malcolm 2023-08-29) set number of workers
    # --num-workers 0
)

set -x

fairseq-train "${FAIRSEQ_ARGS[@]}"

set +x

