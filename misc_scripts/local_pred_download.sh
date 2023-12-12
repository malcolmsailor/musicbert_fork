#!/bin/bash

hostname="transfer-grace.ycrc.yale.edu"
ping -c 1 "$hostname" &>/dev/null

if [ $? -ne 0 ]; then
    echo "$hostname is not reachable"
    exit 42
fi

run_names=("$@")

if [ "$#" -lt 1 ]; then
    echo 'Usage: bash local_pred_download.sh [run_name1 run_name2 ...]'
    exit 1
fi

local_folder=~/output/musicbert/saved_predictions

# joined_run_names=""
# for run_name in "${run_names[@]}"; do
#     joined_run_names+="${run_name},"
# done

# # Remove trailing comma:
# joined_run_names=${joined_run_names%,}

# set -e
# set -x
# echo rsync -avun ms3682@transfer-grace.ycrc.yale.edu:project/saved_predictions/musicbert_fork/{30233767,30233768} ~/tmp/rsync_test
# rsync -avun ms3682@transfer-grace.ycrc.yale.edu:project/saved_predictions/musicbert_fork/{${joined_run_names}} ~/tmp/rsync_test
# set +x

remote_dirs=()

for run_name in "${run_names[@]}"; do
    remote_dirs+=("ms3682@transfer-grace.ycrc.yale.edu:project/saved_predictions/musicbert_fork/${run_name}")
done

set -e
set -x
rsync --archive --compress --verbose "${remote_dirs[@]}" "${local_folder}"
set +x

# rsync --archive --compress --verbose

# remote_dirs=()

# for run_name in "${run_names[@]}"; do
#     remote_dirs+=("project/saved_predictions/musicbert_fork/${run_name}")
# done

# sftp "ms3682@transfer-grace.ycrc.yale.edu" <<EOF
# $(for i in "${!remote_dirs[@]}"; do
#     echo "get -R ${remote_dirs[i]} ${local_folder}"
# done)
# EOF
