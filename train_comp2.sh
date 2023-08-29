echo This script was based on https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.custom_classification.md
echo I created it when debugging my composer classifier. I think it is now obsolete.
exit 1
# TOTAL_NUM_UPDATES=4  
# WARMUP_UPDATES=1      
# LR=0.0005          # Borrowed from musicbert
# HEAD_NAME=composer_head     # Custom name for the classification head.

# # TODO: (Malcolm 2023-08-29) make this a parameter
# NUM_CLASSES=7           # Number of classes for the classification task.
# MAX_SENTENCES=2         


# MAX_POSITIONS=8192
# # NB BATCH_SIZE is only used in the UPDATE_FREQ calculation below; the actual batch size
# #   to fairseq-train is set by MAX_SENTENCES arg
# BATCH_SIZE=64  # Effective batch size.
# MAX_SENTENCES=4  # Actual batch size.

# if command -v nvidia-smi > /dev/null 2>&1 ;
# then
#     N_GPU_LOCAL=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#     CPU_FLAG=""
# else
#     N_GPU_LOCAL=0
#     CPU_FLAG=--cpu
# fi

# UPDATE_FREQ_DENOM=$((N_GPU_LOCAL>1 ? N_GPU_LOCAL : 1))
# UPDATE_FREQ=$((${BATCH_SIZE} / ${MAX_SENTENCES} / ${UPDATE_FREQ_DENOM}))
# WANDB_PROJECT=TODO

# LOG_INTERVAL=1 # TODO restore a higher value
# CHECKPOINT_METRIC=accuracy # TODO: (Malcolm 2023-08-29) use f1?

# USER_DIR=/Users/malcolm/google_drive/python/third_party/musicbert_fork/musicbert
# DATA_BIN_DIR=/Users/malcolm/tmp/composer_classification_data_bin
# # Removed:
# # TODO check these
# # --init-token 0 --separator-token 2 \

# set -x
# echo fairseq-train ${DATA_BIN_DIR} \
#     ${CPU_FLAG} \
#     --wandb-project ${WANDB_PROJECT} \
#     --user-dir ${USER_DIR} \
#     --max-positions 8192 \
#     --batch-size $MAX_SENTENCES \
#     --max-tokens 4400 \
#     --task sentence_prediction \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --required-batch-size-multiple 1 \
#     --arch musicbert_test \
#     --criterion sentence_prediction \
#     --classification-head-name $HEAD_NAME \
#     --num-classes $NUM_CLASSES \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
#     --clip-norm 0.0 \
#     --lr-scheduler polynomial_decay --lr $LR \
#     --log-format simple \
#     --log-interval ${LOG_INTERVAL} \
#     --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
#     --max-update ${TOTAL_NUM_UPDATES} \
#     --best-checkpoint-metric ${CHECKPOINT_METRIC} --maximize-best-checkpoint-metric \
#     --shorten-method "truncate" \
#     --find-unused-parameters \
#     --update-freq ${UPDATE_FREQ}
# set +x
