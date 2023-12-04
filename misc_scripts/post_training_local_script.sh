#!/bin/bash

# Created by giving following prompt to ChatGPT:
# Can you write a bash script that takes a SLURM_ID as argument and performs the following steps:

# 1. download predictions from grace: `bash /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/local_pred_download.sh SLURM_ID`
# 2. see synced metrics (gives a fairer comparison with AugmentedNet etc):
#     1. initialize `music_df` venv in `/Users/malcolm/venvs/music_df`
#     1. `bash /Users/malcolm/google_drive/python/malmus/music_df/user_scripts/musicbert_metrics.sh SLURM_ID test`
#     2. `bash /Users/malcolm/google_drive/python/malmus/music_df/user_scripts/musicbert_synced_metrics.sh ~/output/musicbert_collated_predictions/SLURM_ID/test`

# Check if a SLURM_ID is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 SLURM_ID"
    exit 1
fi

set -e

SLURM_ID=$1

# Step 1: Download predictions
echo "Downloading predictions for SLURM_ID: $SLURM_ID"
set -x
bash /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/local_pred_download.sh $SLURM_ID
set +x

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Download failed for SLURM_ID: $SLURM_ID"
    exit 1
fi

# Step 2: See synced metrics
# 2.1 Initialize music_df venv
echo "Initializing music_df venv..."
source /Users/malcolm/venvs/music_df/bin/activate

# Check if venv activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate music_df venv."
    exit 1
fi

# 2.2 Run musicbert_metrics.sh
echo "Running musicbert_metrics.sh for SLURM_ID: $SLURM_ID"
set -x
bash /Users/malcolm/google_drive/python/malmus/music_df/user_scripts/musicbert_metrics.sh $SLURM_ID test
set +x

# Check if musicbert_metrics was successful
if [ $? -ne 0 ]; then
    echo "musicbert_metrics.sh failed for SLURM_ID: $SLURM_ID"
    exit 1
fi

# 2.3 Run musicbert_synced_metrics.sh
echo "Running musicbert_synced_metrics.sh for SLURM_ID: $SLURM_ID"
set -x
bash /Users/malcolm/google_drive/python/malmus/music_df/user_scripts/musicbert_synced_metrics.sh ~/output/musicbert_collated_predictions/$SLURM_ID/test
set +x

# Check if musicbert_synced_metrics was successful
if [ $? -ne 0 ]; then
    echo "musicbert_synced_metrics.sh failed for SLURM_ID: $SLURM_ID"
    exit 1
fi

echo "All operations completed successfully for SLURM_ID: $SLURM_ID"

# Deactivate the virtual environment
deactivate
