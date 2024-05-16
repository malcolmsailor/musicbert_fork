import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import OmegaConf
from zipfile import ZipFile


@dataclass
class Config:
    input_folder: str
    feature_names: List[str]
    output_folder: str
    conditioning: Optional[str] = None
    external_conditioning: Optional[str] = None
    inputs_name: str = "input0"
    # --conditioning is only implemented for --multitask, which expects
    #   the labels in numbered directories `label0/`, `label1/`, etc.
    #   If we want to condition a single task, then we can do
    #   `force_multitask=True``
    force_multitask: bool = False


def make_links(feature, output_subfolder, config):
    src_dir = os.path.join(config.input_folder, feature)
    assert os.path.isdir(src_dir)
    dst_dir = os.path.join(config.output_folder, output_subfolder)

    print(f"Linking {dst_dir} -> {src_dir}")
    os.symlink(src_dir, dst_dir)


def make_metadata_links(config):
    assert config.input_folder.rstrip(os.path.sep).endswith("_bin")
    raw_data_dir = config.input_folder.rstrip(os.path.sep)[:-4] + "_raw"
    zipped_data = False
    try:
        assert os.path.exists(raw_data_dir)
    except AssertionError:
        zip_data = raw_data_dir + ".zip"
        assert os.path.exists(zip_data), f"neither {raw_data_dir} or {zip_data} exists"
        zipped_data = True

    if not zipped_data:
        for split in ("train", "valid", "test"):
            metadata_basename = f"metadata_{split}.txt"
            src = os.path.join(raw_data_dir, metadata_basename)
            if os.path.exists(src):
                dst = os.path.join(config.output_folder, metadata_basename)
                print(f"Linking {src} -> {dst}")
                os.symlink(src, dst)

    else:
        zip_file = ZipFile(zip_data)
        for split in ("train", "valid", "test"):
            metadata_basename = f"metadata_{split}.txt"
            src = os.path.join(os.path.basename(raw_data_dir), metadata_basename)
            if src in zip_file.namelist():
                with zip_file.open(src) as src_file:
                    dst = os.path.join(config.output_folder, metadata_basename)
                    with open(dst, "wb") as dst_file:
                        dst_file.write(src_file.read())
                    print(f"Copied {zip_file}::{src} -> {dst}")


def make_external_conditioning_links(config: Config):
    assert config.external_conditioning
    assert os.path.isdir(config.external_conditioning)
    dst_dir = os.path.join(config.output_folder, "conditioning")
    print(f"Linking {dst_dir} -> {config.external_conditioning}")
    os.symlink(config.external_conditioning, dst_dir)


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

    if (not config.force_multitask) and (len(config.feature_names) == 1):
        make_links(config.feature_names[0], "label", config)
    else:
        for i, feature in enumerate(config.feature_names):
            make_links(feature, f"label{i}", config)

    assert not (config.conditioning and config.external_conditioning)

    if config.conditioning:
        make_links(config.conditioning, "conditioning", config)
    elif config.external_conditioning:
        make_external_conditioning_links(config)

    make_metadata_links(config)


if __name__ == "__main__":
    main()
