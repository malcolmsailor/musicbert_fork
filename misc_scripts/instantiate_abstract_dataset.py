import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import OmegaConf


@dataclass
class Config:
    input_folder: str
    feature_names: List[str]
    output_folder: str
    conditioning: Optional[str] = None
    inputs_name: str = "input0"


def make_links(feature, output_subfolder, config):
    src_dir = os.path.join(config.input_folder, feature)
    assert os.path.isdir(src_dir)
    dst_dir = os.path.join(config.output_folder, output_subfolder)

    print(f"Linking {dst_dir} -> {src_dir}")
    os.symlink(src_dir, dst_dir)


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    assert config.output_folder.endswith("_bin")
    assert not os.path.exists(config.output_folder)
    os.makedirs(config.output_folder)

    with open(os.path.join(config.output_folder, "target_names.json"), "w") as outf:
        json.dump(list(config.feature_names), outf)
        print(f"Wrote {os.path.join(config.output_folder, 'target_names.json')}")

    make_links(config.inputs_name, "input0", config)

    if len(config.feature_names) == 1:
        make_links(config.feature_names[0], "label", config)
    else:
        for i, feature in enumerate(config.feature_names):
            make_links(feature, f"label{i}", config)

    if config.conditioning:
        make_links(config.conditioning, "conditioning", config)


if __name__ == "__main__":
    main()
