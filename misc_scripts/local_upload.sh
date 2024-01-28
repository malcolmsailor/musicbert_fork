#!/bin/bash

echo DEPRECATED IN FAVOR OF VERSION IN dissertation_code

data_folder=$1

if [[ -z $data_folder ]]; then
    echo 'Usage: bash local_upload.sh [data_folder]'
    exit 1
fi

cd "${data_folder}/.."
data_folder_dirname=$(dirname "${data_folder}")

data_folder_basename=$(basename "${data_folder}")

if [[ -e "${data_folder_basename}".zip ]]; then
    set -x
    trash "${data_folder_basename}".zip
    set +x
fi

set -e
set -x
zip -r "${data_folder%/}".zip "${data_folder_basename}"
scp -r "${data_folder%/}".zip ms3682@transfer-grace.ycrc.yale.edu:project/raw_data/
ssh ms3682@grace.ycrc.yale.edu "cd project/raw_data; unzip ${data_folder_basename%/}.zip; rm ${data_folder_basename%/}.zip"
set +x
