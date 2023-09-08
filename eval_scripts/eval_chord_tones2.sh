DATA_BIN_DIR=/Users/malcolm/output/fairseq/chord_tones_seqs_bin


# fairseq will extend the vocabulary to a nice size by including words like `madeupword0000`
# We don't want to include those in the output classes
NUM_CLASSES=$(cat "${DATA_BIN_DIR}"/label/dict.txt | grep -v -E "madeupword[0-9]{4}" | wc -l)

python $(dirname "$0")/eval_chord_tones2.py \
    "${DATA_BIN_DIR}" \
    --user-dir $(dirname "$0")/../musicbert \
    --num-classes ${NUM_CLASSES} \
    --restore-file /Volumes/Reicha/large_checkpoints/musicbert/checkpoint_last_musicbert_base.pt \
    --max-positions 8192 \
    --criterion sequence_tagging \
    --compound-token-ratio 8 \
    "${@}"
    # --arch musicbert_test
    # --arch base \

set +x
