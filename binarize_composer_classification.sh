

PREFIX=$1
[[ -z "$PREFIX" ]] && { echo "PREFIX is empty" ; exit 1; }
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }

DATA_RAW=${PREFIX}_data_raw
DATA_BIN=${PREFIX}_data_bin

WORKERS=$2
if [[ -z "${WORKERS}" ]]
then
  WORKERS=24
fi

echo Number of workers: ${WORKERS}

set -x
fairseq-preprocess \
    --only-source \
    --trainpref ${PREFIX}_data_raw/midi_train.txt \
    --validpref ${PREFIX}_data_raw/midi_valid.txt \
    --testpref ${PREFIX}_data_raw/midi_test.txt \
    --destdir ${PREFIX}_data_bin/input0 \
    --srcdict ${PREFIX}_data_raw/dict.input.txt \
    --workers $WORKERS
fairseq-preprocess \
    --only-source \
    --trainpref ${PREFIX}_data_raw/targets_train.txt \
    --validpref ${PREFIX}_data_raw/targets_valid.txt \
    --testpref ${PREFIX}_data_raw/targets_test.txt \
    --destdir ${PREFIX}_data_bin/label \
    --srcdict ${PREFIX}_data_raw/dict.targets.txt \
    --workers $WORKERS
set +x
