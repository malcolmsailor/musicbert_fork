from collections import defaultdict
from itertools import chain
import os
import re
import shutil
import subprocess
import glob

import sys
from dataclasses import dataclass, field
from omegaconf import OmegaConf


@dataclass
class Config:
    input_folder: str
    workers: int = 24
    inputs_name: str = "midi"
    overwrite: bool = False
    truncate_vocab: dict = field(default_factory=dict)

    def __post_init__(self):
        for key, val in self.truncate_vocab.items():
            assert isinstance(val, int)
            # So that " ".join(command) will not fail, we need to cast to string
            self.truncate_vocab[key] = str(val)


conf = OmegaConf.from_cli(sys.argv[1:])
config = Config(**conf)  # type:ignore

assert re.search(r"_raw/?$", config.input_folder)

output_folder = config.input_folder.rstrip(os.path.sep)[:-4] + "_bin"

# assert not os.path.exists(output_folder)

assert config.workers > 0

files = [
    os.path.basename(f)
    for f in chain(
        glob.glob(os.path.join(config.input_folder, "*_train.txt")),
        glob.glob(os.path.join(config.input_folder, "*_valid.txt")),
        glob.glob(os.path.join(config.input_folder, "*_test.txt")),
    )
]

# Exclude metadata files
files = [f for f in files if not f.startswith("metadata_")]

srcdict_path = os.path.join(config.input_folder, "dict.input.txt")
assert os.path.exists(srcdict_path)

inputs = set()
keyed_features = defaultdict(set)
for f in files:
    feature = f.rsplit("_", maxsplit=1)[0]
    if feature == config.inputs_name:
        inputs.add(f)
    else:
        keyed_features[feature].add(f)

assert inputs


def do_feature(feature, files, output_subfolder, src_dict=None, truncate_vocab=None):
    destdir = os.path.join(output_folder, output_subfolder)
    if os.path.exists(destdir):
        if config.overwrite:
            shutil.rmtree(destdir)
        else:
            print(f"{destdir} exists, skipping")
            return
    command = ["fairseq-preprocess", "--only-source"]
    if f"{feature}_train.txt" in files:
        command.extend(
            ["--trainpref", os.path.join(config.input_folder, f"{feature}_train.txt")]
        )
    if f"{feature}_valid.txt" in files:
        command.extend(
            ["--validpref", os.path.join(config.input_folder, f"{feature}_valid.txt")]
        )
    if f"{feature}_test.txt" in files:
        command.extend(
            ["--testpref", os.path.join(config.input_folder, f"{feature}_test.txt")]
        )
    command.extend(
        [
            "--destdir",
            os.path.join(output_folder, output_subfolder),
            "--workers",
            str(config.workers),
        ]
    )

    assert not (src_dict and truncate_vocab)

    if src_dict:
        command.extend(["--srcdict", src_dict])
    elif truncate_vocab:
        command.extend(["--nwordssrc", truncate_vocab])
    else:
        # Check if there is an existing dict for this feature
        src_dict_path = os.path.join(config.input_folder, f"dict.{feature}.txt")
        if os.path.exists(src_dict_path):
            command.extend(["--srcdict", src_dict_path])

    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


do_feature("midi", inputs, "input0", srcdict_path)

for key, files in keyed_features.items():
    do_feature(key, files, key, truncate_vocab=config.truncate_vocab.get(key))
