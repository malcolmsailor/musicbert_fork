TOTAL_UPDATES=125000
WARMUP_UPDATES=25000

PEAK_LR=0.0005 # Borrowed from musicbert

# NB in musicbert scripts, BATCH_SIZE is only used in the UPDATE_FREQ calculation below;
#   the actual batch size to fairseq-train is set by MAX_SENTENCES arg
BATCH_SIZE=64
MAX_SENTENCES=4

if command -v nvidia-smi > /dev/null 2>&1 ;
then
    N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    N_GPU_LOCAL=0
fi

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

FAIRSEQ_ARGS=(
    ${DATA_BIN_DIR}
    --user-dir ${USER_DIR}
    --wandb-project ${WANDB_PROJECT}
    --task composer_classification # TODO Do we need to add --criterion
    --arch ${NN_ARCH}
    --batch-size $MAX_SENTENCES 
    # --update-freq $UPDATE_FREQ # TODO

    # TODO do we need to specify num-classes
    # TODO do we need to specify classification-head-name

    # TODO restore from checkpoint
    # --restore-file $MUSICBERT_PATH \
    
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
    # --shorten-method 'truncate' TODO investigate
    --no-epoch-checkpoints
    --find-unused-parameters
    --best-checkpoint-metric f1_score_micro 
    --maximize-best-checkpoint-metric

    # Args from musicbert not (yet?) using:
    # --max-positions $MAX_POSITIONS \
    # --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
    # --reset-optimizer --reset-dataloader --reset-meters \
    # --required-batch-size-multiple 1 \
    # --init-token 0 --separator-token 2 \

    # Why does musicbert set num workesr to 0???
    # --num-workers 0 \
)

set -x
fairseq-train "${FAIRSEQ_ARGS[@]}"
set +x

# set -x
# fairseq-train ${DATA_BIN_DIR} \
#     --user-dir ${USER_DIR} \
#     --wandb-project ${WANDB_PROJECT} \
#     --task composer_classification `# TODO Do we need to add --criterion` \
#     --optimizer adam --adam-betas '(0.9,0.98)' `# Copied from train_genre` \
#     --adam-eps 1e-6 --clip-norm 0.0 `# Copied from train_genre` \
#     --lr-scheduler polynomial_decay --lr ${PEAK_LR} \
#     --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
#     --arch ${NN_ARCH}
# set +x
