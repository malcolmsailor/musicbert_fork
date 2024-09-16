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
    echo "Second argument (num workers) is required"
    exit 1
fi

if [[ $WORKERS -le 0 ]]; then
    echo "Number of workers must be greater than 0"
    exit 1
fi

echo Number of workers: ${WORKERS}

# Set vocabulary thresholds if provided

# Check if the number of remaining arguments is odd
if (($# % 2 != 0)); then
    echo "Error: Odd number of arguments. We require an even number of arguments."
    echo "    Set vocabulary thresholds as 'idx threshold' pairs"
    exit 1
fi

# Declare an associative array
declare -A thresholds

# Skip the first two arguments ($1 and $2) and start from $3
idx=3

# Loop over the remaining arguments, pairing them as key/value
while [[ $idx -le $# ]]; do
    key=${!idx}             # Get the key
    next_index=$((idx + 1)) # Calculate the next index
    value=${!next_index}    # Get the value using the next index
    thresholds[$key]=$value # Store in the associative array

    idx=$((idx + 2)) # Move to the next key/value pair
done

target_i=0
((target_i++))

if [[ -v thresholds[$target_i] ]]; then
    echo "$target_i: ${thresholds[$target_i]}"
fi

exit

# # Print the contents of the associative array (for verification)
# for key in "${!thresholds[@]}"; do
#     echo "$key: ${thresholds[$key]}"
# done

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

        # Apply threshold if provided
        if [[ -v thresholds[$target_i] ]]; then
            # fairseq-preprocess has two arguments, --thresholdsrc and --thresholdtgt
            #   I'm not completely sure whether it thinks we're processing source or
            #   target here (of course, in actuality we are processing target, but
            #   not sure if fairseq-preprocess knows that). Perhaps easiest to
            #   just set both.
            command+=" --thresholdtgt ${thresholds[$target_i]}"
            command+=" --thresholdsrc ${thresholds[$target_i]}"
        fi

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

# Conditioning

conditioning=""
# Check if the training file exists
if [ -f "${DATA_RAW}/conditioning_train.txt" ]; then
    conditioning+=" --trainpref ${DATA_RAW}/conditioning_train.txt"
fi

# Check if the validation file exists
if [ -f "${DATA_RAW}/conditioning_valid.txt" ]; then
    conditioning+=" --validpref ${DATA_RAW}/conditioning_valid.txt"
fi

# Check if the test file exists
if [ -f "${DATA_RAW}/conditioning_test.txt" ]; then
    conditioning+=" --testpref ${DATA_RAW}/conditioning_test.txt"
fi

# Conditioning
if [[ ! -z "$conditioning" ]]; then
    command="fairseq-preprocess --only-source"
    command+=" $conditioning"
    # Continue building the command
    command+=" --destdir ${DATA_BIN}/conditioning --workers $WORKERS"
    set -x
    eval $command
    set +x
fi
