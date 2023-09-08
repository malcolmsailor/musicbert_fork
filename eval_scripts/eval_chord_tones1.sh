DATA_BIN_DIR=/Users/malcolm/output/fairseq/chord_tones_seqs_bin

NUM_CLASSES=$(cat "${DATA_BIN_DIR}"/label/dict.txt | grep -v -E "madeupword[0-9]{4}" | wc -l)

set -x

python $(dirname "$0")/eval_chord_tones.py \
    --data-dir "${DATA_BIN_DIR}" \
    --checkpoint /Volumes/Reicha/large_checkpoints/musicbert/checkpoint_last_musicbert_base.pt \
    "${@}"
    # --arch musicbert_test
    # --arch base \

set +x
