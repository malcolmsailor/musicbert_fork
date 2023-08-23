# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import json
import os
import random
import sys
import zipfile
from functools import partial
from multiprocessing import Manager, Pool

from sklearn.model_selection import StratifiedKFold

import preprocess

SUBSET = input("subset: ")
RAW_DATA_DIR = SUBSET + "_data_raw"
if os.path.exists(RAW_DATA_DIR):
    print("Output path {} already exists!".format(RAW_DATA_DIR))
    sys.exit(0)
DATA_PATH = os.path.join(os.environ["DATASETS_DIR"], "lmd", "lmd_full.zip")
N_FOLDS = 5
N_TIMES = 4  # sample train set multiple times
MAX_LENGTH = int(input("sequence length: "))
preprocess.SAMPLE_LEN_MAX = MAX_LENGTH
preprocess.DEDUPLICATE = False
preprocess.DATA_ZIP = zipfile.ZipFile(DATA_PATH)
FOLD_MAP = dict()
MULTIPROCESS = True
if MULTIPROCESS:
    MANAGER = Manager()
    MIDI_DICT = MANAGER.dict()
    ALL_DATA = MANAGER.list()
else:
    MIDI_DICT = {}
    ALL_DATA = []

POOL_NUM = 24

GENRE_MAP_JSON = os.path.join(os.environ["DATASETS_DIR"], "midi_genre_map.json")
LABELS = dict()

with open(GENRE_MAP_JSON) as f:
    for s in json.load(f)[SUBSET].items():
        # Looks like Labels[Genre] = tuple(songs)
        LABELS[s[0]] = tuple(sorted(set(i.strip().replace(" ", "-") for i in s[1])))


def get_id(file_name):
    # Gets basename without extension
    return file_name.split("/")[-1].split(".")[0]


def get_fold(file_name):
    return FOLD_MAP[get_id(file_name)]


def get_sample(output_str_list):
    max_len = max(len(s.split()) for s in output_str_list)
    return random.choice([s for s in output_str_list if len(s.split()) == max_len])


def new_writer(file_name, output_str_list):
    if len(output_str_list) > 0:
        ALL_DATA.append(
            (file_name, tuple(get_sample(output_str_list) for _ in range(N_TIMES)))
        )


preprocess.writer = new_writer


DATA_PATH = os.path.join(os.environ["DATASETS_DIR"], "lmd", "lmd_full.zip")


def main():
    DATA_ZIP = zipfile.ZipFile(DATA_PATH, "r")

    os.system("mkdir -p {}".format(RAW_DATA_DIR))
    file_list = [
        file_name
        for file_name in DATA_ZIP.namelist()
        if file_name[-4:].lower() == ".mid" or file_name[-5:].lower() == ".midi"
    ]
    file_list = [file_name for file_name in file_list if get_id(file_name) in LABELS]
    random.shuffle(file_list)
    label_list = ["+".join(LABELS[get_id(file_name)]) for file_name in file_list]
    fold_index = 0
    encode_f = partial(
        preprocess.encode_file_with_error_handling,
        midi_dict=MIDI_DICT,
        zip_path=DATA_PATH,
        output_file=TODO,
    )
    for train_index, test_index in StratifiedKFold(N_FOLDS).split(
        file_list, label_list
    ):
        for i in test_index:
            FOLD_MAP[get_id(file_list[i])] = fold_index
        fold_index += 1
    with Pool(POOL_NUM) as p:
        list(p.imap_unordered(preprocess.encode_file_with_error_handling, file_list))
    random.shuffle(ALL_DATA)
    print(
        "{}/{} ({:.2f}%)".format(
            len(ALL_DATA), len(file_list), len(ALL_DATA) / len(file_list) * 100
        )
    )
    for fold in range(N_FOLDS):
        os.system("mkdir -p {}/{}".format(RAW_DATA_DIR, fold))
        preprocess.gen_dictionary("{}/{}/dict.txt".format(RAW_DATA_DIR, fold))
        for cur_split in ["train", "test"]:
            output_path_prefix = "{}/{}/{}".format(RAW_DATA_DIR, fold, cur_split)
            with open(output_path_prefix + ".txt", "w") as f_txt:
                with open(output_path_prefix + ".label", "w") as f_label:
                    with open(output_path_prefix + ".id", "w") as f_id:
                        count = 0
                        for file_name, output_str_list in ALL_DATA:
                            if (
                                cur_split == "train" and fold != get_fold(file_name)
                            ) or (cur_split == "test" and fold == get_fold(file_name)):
                                for i in range(N_TIMES if cur_split == "train" else 1):
                                    f_txt.write(output_str_list[i] + "\n")
                                    f_label.write(
                                        " ".join(LABELS[get_id(file_name)]) + "\n"
                                    )
                                    f_id.write(get_id(file_name) + "\n")
                                    count += 1
                        print(fold, cur_split, count)


if __name__ == "__main__":
    main()
