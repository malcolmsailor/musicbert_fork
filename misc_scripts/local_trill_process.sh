# Check if a SLURM_ID is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SLURM_IDS"
    exit 1
fi

SLURM_IDS=("$@")

# Step 1: Download predictions
# echo "Downloading predictions for SLURM_IDS: ${SLURM_IDS[@]}"
# set -x
# bash /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/local_unlabeled_pred_download.sh ${SLURM_IDS[@]}
# set +x

# # Check if download was successful
# return_code=$?

# if [ $return_code -eq 42 ]; then
#     echo "Host is not available, skipping rsync"
#     read -p "Do you want to continue (i.e., if local files already downloaded)? (y/n) " response

#     # Check the response
#     if [[ $response =~ ^[Yy]$ ]]; then
#         echo "Proceeding..."
#     else
#         echo "Exiting..."
#         exit 1
#     fi
# else
#     if [ $return_code -ne 0 ]; then
#         echo "Download failed for SLURM_IDS: ${SLURM_IDS[@]}"
#         exit 1
#     fi
# fi

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
INPUT_BASE_FOLDER=~/output/musicbert_unlabeled/saved_predictions
OUTPUT_BASE_FOLDER=~/output/musicbert_unlabeled/collated_predictions
LABEL_SCRIPT="${MUSIC_DF_FOLDER}/scripts/label_dfs.py"

TRILLS_FOLDER=~/output/trills/labeled_dfs

for SLURM_ID in ${SLURM_IDS[@]}; do
    output_folder="${OUTPUT_BASE_FOLDER}/${SLURM_ID}/test"
    metadata="${INPUT_BASE_FOLDER}/${SLURM_ID}/test/metadata_test.txt"
    predictions="${INPUT_BASE_FOLDER}/${SLURM_ID}/test/predictions"
    # set -x
    # python "${COLLATE_SCRIPT}" metadata="${metadata}" predictions="${predictions}" \
    #     prediction_file_type=both txt_overlaps=midpoint h5_overlaps=weighted_average \
    #     output_folder="${output_folder}"
    # set +x

    mkdir -p "${output_folder}"

    # for dictionary in "${INPUT_BASE_FOLDER}/${SLURM_ID}"/test/*_dictionary.txt; do
    #     set -x
    #     cp $dictionary "${output_folder}"/$(basename ${dictionary})
    #     set +x
    # done
    trill_folder="${TRILLS_FOLDER}/${SLURM_ID}"
    set -x
    python "${LABEL_SCRIPT}" \
        --config-file "${MUSIC_DF_FOLDER}/scripts/configs/label_trills1.yaml" \
        metadata_path="${output_folder}"/metadata_test.txt \
        labels_path="${output_folder}"/predictions/label.txt \
        output_folder="${trill_folder}"

    set +x
done
exit

echo "Initializing trill_data venv..."
source /Users/malcolm/venvs/trill_data/bin/activate

# Check if venv activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate trill_data venv."
    exit 1
fi

TRILL_DATA_FOLDER=/Users/malcolm/google_drive/python/data_science/trill_data
DETRILLED_FOLDER=~/output/trills/detrilled_dfs
MIDI_FOLDER=~/output/trills/detrilled_midi
for SLURM_ID in ${SLURM_IDS[@]}; do
    set -x
    python "${TRILL_DATA_FOLDER}"/scripts/remove_trills.py \
        "${TRILLS_FOLDER}"/"${SLURM_ID}" \
        "${DETRILLED_FOLDER}"/"${SLURM_ID}"
    python "${MUSIC_DF_FOLDER}"/scripts/csvs_to_midi.py \
        "${DETRILLED_FOLDER}"/"${SLURM_ID}" \
        "${MIDI_FOLDER}"/"${SLURM_ID}"
    set +x

done
