#!/bin/bash

run_name=$1

if [[ -z $run_name ]]; then
    echo 'Usage: bash local_upload.sh [run_name]'
    exit 1
fi

remote_folder=project/saved_predictions/musicbert_fork/"${run_name}"

local_folder=~/output/musicbert/saved_predictions/
set -e
set -x
mkdir -p "${local_folder}"
scp -r ms3682@transfer-grace.ycrc.yale.edu:"${remote_folder}" "${local_folder}"
set +x
