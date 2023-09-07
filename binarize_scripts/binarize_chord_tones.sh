

DATA_RAW=$1
if ! [[ "$DATA_RAW" =~ .*_raw$ ]]
then
    echo First argument must be raw data path ending in _raw
    exit 1
fi

DATA_BIN=${DATA_RAW%_raw}_bin

[[ -d "${DATA_BIN}" ]] && { echo "output directory ${DATA_BIN} already exists" ; exit 1; }


WORKERS=$2
if [[ -z "${WORKERS}" ]]
then
  WORKERS=24
fi

echo Number of workers: ${WORKERS}

set -x
fairseq-preprocess \
    --only-source \
    --trainpref ${DATA_RAW}/midi_train.txt \
    --validpref ${DATA_RAW}/midi_valid.txt \
    --testpref ${DATA_RAW}/midi_test.txt \
    --destdir ${DATA_BIN}/input0 \
    --srcdict ${DATA_RAW}/dict.input.txt \
    --workers $WORKERS
fairseq-preprocess \
    --only-source \
    --trainpref ${DATA_RAW}/targets_train.txt \
    --validpref ${DATA_RAW}/targets_valid.txt \
    --testpref ${DATA_RAW}/targets_test.txt \
    --destdir ${DATA_BIN}/label \
    --workers $WORKERS
# When we read the binarized version, there seems to be an extra token (EOS?) added to
# each row.
set +x

