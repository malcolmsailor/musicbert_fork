#!/bin/bash

hostname="transfer-grace.ycrc.yale.edu"
ping -c 1 "$hostname" &>/dev/null

if [ $? -ne 0 ]; then
    echo "$hostname is not reachable"
    exit 42
fi

run_names=("$@")

if [ "$#" -lt 1 ]; then
    echo 'Usage: bash local_unlabeled_pred_download.sh [run_name1 run_name2 ...]'
    exit 1
fi

# remote_folder=project/saved_predictions/musicbert_unlabeled/"${run_name}"

local_folder=~/output/musicbert_unlabeled/saved_predictions/
remote_dirs=()

for run_name in "${run_names[@]}"; do
    remote_dirs+=("ms3682@transfer-grace.ycrc.yale.edu:project/saved_predictions/musicbert_unlabeled/${run_name}")
done

# mkdir -p "${local_folder}"
# scp -r ms3682@transfer-grace.ycrc.yale.edu:"${remote_folder}" "${local_folder}"
set -e
set -x
rsync --archive --compress --verbose "${remote_dirs[@]}" "${local_folder}"
set +x
