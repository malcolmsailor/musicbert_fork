import re
import sys
from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf
from fairseq.models.roberta import RobertaModel
import os
import torch
import lovely_tensors

lovely_tensors.monkey_patch()

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.append(PARENT_DIR)

import traceback, pdb, sys


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    output_path: str
    inner_states_path: Optional[str] = None
    checkpoint: str = os.path.expanduser(
        "~/output/test_checkpoints/combined_key_musicbert_small.pt"
    )
    data_dir: str = os.path.expanduser("~/output/test_data/combined_key_bin/")
    model_name_or_path: str = PARENT_DIR
    user_dir: str = USER_DIR


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    m = re.match(r".*(?P<arch>musicbert_\w+)\.pt", config.checkpoint)
    assert m
    arch = m.group("arch")

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        # The only way I seem to be able to get musicbert to load is by passing
        #   the (presumably private) _name parameter. There must be another way!
        _name=arch,
        checkpoint_file=config.checkpoint,
        data_name_or_path=config.data_dir,
        user_dir=config.user_dir,
        task="musicbert_sequence_tagging",
    ).model

    musicbert.eval()  # type:ignore

    sample_input = torch.arange(320).reshape(1, -1)

    output, _ = musicbert(
        sample_input,
        return_all_hiddens=False,
        classification_head_name="sequence_tagging_head",
    )  # type:ignore
    output1, x = musicbert(
        sample_input,
        return_all_hiddens=True,
        classification_head_name="sequence_tagging_head",
    )  # type:ignore

    assert (output == output1).all()

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    torch.save(output, config.output_path)
    print(f"Saved {config.output_path}")

    if config.inner_states_path is not None:
        inner_states = torch.stack(x["inner_states"][:-1])
        os.makedirs(os.path.dirname(config.inner_states_path), exist_ok=True)
        torch.save(inner_states, config.inner_states_path)
        print(f"Saved {config.inner_states_path}")


if __name__ == "__main__":
    main()
