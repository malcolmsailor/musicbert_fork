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

command="fairseq-preprocess --only-source"
# Check if the training file exists
if [ -f "${DATA_RAW}/midi_train.txt" ]; then
    command+=" --trainpref ${DATA_RAW}/midi_train.txt"
fi

# Check if the validation file exists
if [ -f "${DATA_RAW}/midi_valid.txt" ]; then
    command+=" --validpref ${DATA_RAW}/midi_valid.txt"
fi

# Check if the test file exists
if [ -f "${DATA_RAW}/midi_test.txt" ]; then
    command+=" --testpref ${DATA_RAW}/midi_test.txt"
fi

# Continue building the command
command+=" --destdir ${DATA_BIN}/input0 --srcdict ${DATA_RAW}/dict.input.txt --workers $WORKERS"

set -x
eval $command
set +x

if [[ -e ${DATA_RAW}/target_names.json ]]; then
    cp ${DATA_RAW}/target_names.json ${DATA_BIN}/target_names.json
fi

if [[ ! -e ${DATA_RAW}/targets_0_train.txt ]]; then
    # Single target
    command="fairseq-preprocess --only-source"
    # Check if the training file exists
    if [ -f "${DATA_RAW}/targets_train.txt" ]; then
        command+=" --trainpref ${DATA_RAW}/targets_train.txt"
    fi

    # Check if the validation file exists
    if [ -f "${DATA_RAW}/targets_valid.txt" ]; then
        command+=" --validpref ${DATA_RAW}/targets_valid.txt"
    fi

    # Check if the test file exists
    if [ -f "${DATA_RAW}/targets_test.txt" ]; then
        command+=" --testpref ${DATA_RAW}/targets_test.txt"
    fi

    # Continue building the command
    command+=" --destdir ${DATA_BIN}/label --workers $WORKERS"

    set -x
    eval $command
    # When we read the binarized version, there seems to be an extra token (EOS?) added to
    # each row.
    set +x
else
    # Multi-target
    target_i=0
    while true; do
        if [[ ! -e ${DATA_RAW}/targets_${target_i}_train.txt ]]; then
            break
        fi
        command="fairseq-preprocess --only-source"
        # Check if the training file exists
        if [ -f "${DATA_RAW}/targets_${target_i}_train.txt" ]; then
            command+=" --trainpref ${DATA_RAW}/targets_${target_i}_train.txt"
        fi

        # Check if the validation file exists
        if [ -f "${DATA_RAW}/targets_${target_i}_valid.txt" ]; then
            command+=" --validpref ${DATA_RAW}/targets_${target_i}_valid.txt"
        fi

        # Check if the test file exists
        if [ -f "${DATA_RAW}/targets_${target_i}_test.txt" ]; then
            command+=" --testpref ${DATA_RAW}/targets_${target_i}_test.txt"
        fi

        # Continue building the command
        command+=" --destdir ${DATA_BIN}/label${target_i} --workers $WORKERS"
        set -x
        eval $command
        # When we read the binarized version, there seems to be an extra token (EOS?) added to
        # each row.
        set +x
        ((target_i++))
    done
fi
