

DATA_RAW=$1
if ! [[ "$DATA_RAW" =~ .*_data_raw$ ]]
then
    echo First argument must be raw data path ending in _raw_data
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
# I'm not sure we actually need to binarize the targets given we are copying the raw
# labels below.
fairseq-preprocess \
    --only-source \
    --trainpref ${DATA_RAW}/targets_train.txt \
    --validpref ${DATA_RAW}/targets_valid.txt \
    --testpref ${DATA_RAW}/targets_test.txt \
    --destdir ${DATA_BIN}/label \
    --srcdict ${DATA_RAW}/dict.targets.txt \
    --workers $WORKERS
# When we read the binarized version, there seems to be an extra token (EOS?) added to
# each row.
# TODO: (Malcolm 2023-08-29) not sure if this is still necessary after switching to
#   new classification task.
cp ${DATA_RAW}/$i/targets_train.txt ${DATA_BIN}/$i/label/train.label
cp ${DATA_RAW}/$i/targets_valid.txt ${DATA_BIN}/$i/label/valid.label
cp ${DATA_RAW}/$i/targets_test.txt ${DATA_BIN}/$i/label/test.label
set +x

