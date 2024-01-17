#!/bin/bash

hostname="transfer-grace.ycrc.yale.edu"
ping -c 1 "$hostname" &>/dev/null

if [ $? -ne 0 ]; then
    echo "$hostname is not reachable"
    exit 42
fi

text_only=false
run_names=()

if [ "$#" -lt 1 ]; then
    echo 'Usage: bash local_unlabeled_pred_download.sh [-t] [run_name1 run_name2 ...]'
    echo '    -t: download .txt predictions only (exclude .h5 files)'
    exit 1
fi

# Loop through the arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -t)
        text_only=true
        shift # Move to next argument
        ;;
    *)                    # Any other argument
        run_names+=("$1") # Add to the array
        shift             # Move to next argument
        ;;
    esac
done

# run_names=("$@")

# remote_folder=project/saved_predictions/musicbert_unlabeled/"${run_name}"

local_folder=~/output/musicbert_unlabeled/saved_predictions/
remote_dirs=()

for run_name in "${run_names[@]}"; do
    remote_dirs+=("ms3682@transfer-grace.ycrc.yale.edu:project/saved_predictions/musicbert_unlabeled/${run_name}")
done

# mkdir -p "${local_folder}"
# scp -r ms3682@transfer-grace.ycrc.yale.edu:"${remote_folder}" "${local_folder}"
set -e
if [ "$text_only" = true ]; then
    set -x
    rsync --exclude='*.h5' --archive --compress --verbose "${remote_dirs[@]}" "${local_folder}"
    set +x
else
    set -x
    rsync --archive --compress --verbose "${remote_dirs[@]}" "${local_folder}"
    set +x
fi
