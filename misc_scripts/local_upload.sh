#!/bin/bash

data_folder=$1

if [[ -z $data_folder ]]; then
    echo 'Usage: bash local_upload.sh [data_folder]'
    exit 1
fi

if [[ -e "${data_folder}".zip ]]; then
    set -x
    trash "${data_folder}".zip
    set +x
fi

set -x
zip -r "${data_folder}".zip "${data_folder}"
scp -r "${data_folder}".zip ms3682@transfer-grace.ycrc.yale.edu:project/raw_data/
set +x
