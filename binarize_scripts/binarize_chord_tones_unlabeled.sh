DATA_RAW=$1

# Trim trailing slash if present
DATA_RAW=$(echo "$DATA_RAW" | sed 's:/*$::')

if ! [[ "$DATA_RAW" =~ .*_raw$ ]]; then
    echo First argument must be raw data path ending in _raw
    exit 1
fi

REF_FOLDER=$2

if [[ -z $2 ]]; then
    echo "Usage binarize_chord_tones_unlabeled.sh [DATA_RAW] [REF_FOLDER] <NUM_WORKERS>"
    exit 1
fi

DATA_BIN=${DATA_RAW%_raw}_bin

[[ -d "${DATA_BIN}" ]] && {
    echo "output directory ${DATA_BIN} already exists"
    exit 1
}

WORKERS=$3
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

if [[ -e ${REF_FOLDER}/target_names.json ]]; then
    cp ${REF_FOLDER}/target_names.json ${DATA_BIN}/target_names.json
fi

for label in "${REF_FOLDER}"/*; do
    if [ $(basename "$label") = "input0" ] || ! [ -d "$label" ]; then
        continue
    fi
    new_label_dir="${DATA_BIN}/$(basename $label)"
    mkdir -p "${new_label_dir}"
    cp "${label}"/dict.txt "${new_label_dir}"/dict.txt
done
set +x
