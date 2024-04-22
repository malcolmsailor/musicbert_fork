# Check if a SLURM_ID is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SLURM_IDS"
    exit 1
fi

SLURM_IDS=("$@")

set -e

# Step 2: See synced metrics
# 2.1 Initialize music_df venv
echo "Initializing music_df venv..."
source /Users/malcolm/venvs/music_df/bin/activate

# Check if venv activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate music_df venv."
    exit 1
fi

MUSIC_DF_FOLDER="/Users/malcolm/google_drive/python/malmus/music_df"

COLLATE_SCRIPT="${MUSIC_DF_FOLDER}/scripts/collate_predictions.py"
LABELED_INPUT_BASE_FOLDER=/Volumes/Zarebski/musicbert/saved_predictions
LABELED_OUTPUT_BASE_FOLDER=/Volumes/Zarebski/musicbert/collated_predictions
UNLABELED_INPUT_BASE_FOLDER=/Volumes/Zarebski/musicbert_unlabeled/saved_predictions
UNLABELED_OUTPUT_BASE_FOLDER=/Volumes/Zarebski/musicbert_unlabeled/collated_predictions

LABEL_SCRIPT="${MUSIC_DF_FOLDER}/scripts/label_dfs.py"

LABELED_DF_FOLDER=~/output/quantized/labeled_dfs

for SLURM_ID in ${SLURM_IDS[@]}; do
    if [ -d "$UNLABELED_INPUT_BASE_FOLDER"/"${SLURM_ID}" ]; then
        input_base_folder="$UNLABELED_INPUT_BASE_FOLDER"
        output_base_folder="$UNLABELED_OUTPUT_BASE_FOLDER"
    else
        input_base_folder="$LABELED_INPUT_BASE_FOLDER"
        output_base_folder="$LABELED_OUTPUT_BASE_FOLDER"
    fi

    output_folder="${output_base_folder}/${SLURM_ID}/test"
    metadata="${input_base_folder}/${SLURM_ID}/test/metadata_test.txt"
    predictions="${input_base_folder}/${SLURM_ID}/test/predictions"

    set -x
    python "${COLLATE_SCRIPT}" metadata="${metadata}" predictions="${predictions}" \
        prediction_file_type=both txt_overlaps=midpoint h5_overlaps=weighted_average \
        output_folder="${output_folder}"
    set +x

    mkdir -p "${output_folder}"

    for dictionary in "${input_base_folder}/${SLURM_ID}"/test/*_dictionary.txt; do
        set -x
        cp $dictionary "${output_folder}"/$(basename ${dictionary})
        set +x
    done

    quantized_folder="${LABELED_DF_FOLDER}/${SLURM_ID}"

    set -x
    python "${LABEL_SCRIPT}" \
        --config-file "${MUSIC_DF_FOLDER}/scripts/configs/label_quantize_ticks1.yaml" \
        metadata_path="${output_folder}"/metadata_test.txt \
        labels_path=["${output_folder}"/predictions/onset_delta_target_ticks.txt,"${output_folder}"/predictions/release_delta_target_ticks.txt] \
        output_folder="${quantized_folder}" debug=True

    set +x
done

echo "Initializing quantize_data venv..."
source /Users/malcolm/venvs/quantize_data/bin/activate

# Check if venv activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate quantize_data venv."
    exit 1
fi

QUANTIZE_DATA_FOLDER=/Users/malcolm/google_drive/python/data_science/quantize_data
QUANTIZED_DF_FOLDER=~/output/quantized/quantized_dfs
MIDI_FOLDER=~/output/quantized/quantized_midi

for SLURM_ID in ${SLURM_IDS[@]}; do

    set -x
    python "${QUANTIZE_DATA_FOLDER}"/scripts/apply_delta_ticks_quantization.py \
        "${LABELED_DF_FOLDER}"/"${SLURM_ID}" \
        "${QUANTIZED_DF_FOLDER}"/"${SLURM_ID}" --debug
    python "${MUSIC_DF_FOLDER}"/scripts/csvs_to_midi.py \
        "${QUANTIZED_DF_FOLDER}"/"${SLURM_ID}" \
        "${MIDI_FOLDER}"/"${SLURM_ID}"
    set +x

done
