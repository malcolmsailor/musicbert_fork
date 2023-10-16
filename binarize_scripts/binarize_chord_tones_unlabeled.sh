DATA_RAW=$1

# Trim trailing slash if present
DATA_RAW=$(echo "$DATA_RAW" | sed 's:/*$::')

if ! [[ "$DATA_RAW" =~ .*_raw$ ]]; then
    echo First argument must be raw data path ending in _raw
    exit 1
fi

DATA_BIN=${DATA_RAW%_raw}_bin

[[ -d "${DATA_BIN}" ]] && {
    echo "output directory ${DATA_BIN} already exists"
    exit 1
}

WORKERS=$2
if [[ -z "${WORKERS}" ]]; then
    WORKERS=24
fi

echo Number of workers: ${WORKERS}

set -x
fairseq-preprocess \
    --only-source \
    --testpref ${DATA_RAW}/midi_test.txt \
    --destdir ${DATA_BIN}/input0 \
    --srcdict ${DATA_RAW}/dict.input.txt \
    --workers $WORKERS
set +x
