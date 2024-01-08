import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
import logging

import h5py
import numpy as np
import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.roberta import RobertaModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

sys.path.append(PARENT_DIR)

# from musicbert._musicbert import MusicBERTModel

USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="assumed to end in '_bin' and have an equivalent ending in '_raw' that contains 'metadata_test.txt'",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="test", choices=("test", "valid", "train"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--compound-token-ratio", type=int, default=8)
    parser.add_argument("--msdebug", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir, checkpoint, output_folder = (
        args.data_dir,
        args.checkpoint,
        args.output_folder,
    )
    if args.msdebug:
        import pdb
        import sys
        import traceback

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    if os.path.exists(output_folder):
        raise ValueError(f"Output folder {output_folder} already exists")

    assert data_dir.rstrip(os.path.sep).endswith("_bin")
    raw_data_dir = data_dir.rstrip(os.path.sep)[:-4] + "_raw"
    assert os.path.exists(raw_data_dir)

    with open(os.path.join(data_dir, "target_names.json"), "r") as inf:
        target_names = json.load(inf)

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir,
        user_dir=USER_DIR,
        task="musicbert_multitarget_sequence_tagging",
    )

    musicbert.task.load_dataset(args.dataset)
    dataset = musicbert.task.datasets[args.dataset]

    if torch.cuda.is_available():
        musicbert.cuda()

    musicbert.eval()

    n_examples = len(dataset)
    if args.max_examples is not None:
        n_examples = min(args.max_examples, n_examples)

    os.makedirs(output_folder, exist_ok=False)
    os.makedirs(os.path.join(output_folder, "predictions"), exist_ok=False)

    outfs = {}
    out_hdfs: dict[str, h5py.File] = {}

    label_dictionaries: dict[str, Dictionary] = {}
    for i, target_name in enumerate(target_names):
        outfs[target_name] = open(
            os.path.join(output_folder, "predictions", f"{target_name}.txt"), "w"
        )
        dictionary = musicbert.task.label_dictionaries[i]
        label_dictionaries[target_name] = dictionary
        dictionary.save(os.path.join(output_folder, f"{target_name}_dictionary.txt"))
        out_hdfs[target_name] = h5py.File(
            os.path.join(output_folder, "predictions", f"{target_name}.h5"), "w"
        )

    try:
        for i in range(0, n_examples, args.batch_size):
            samples = [
                dataset[j] for j in range(i, min(n_examples, i + args.batch_size))
            ]
            batch = dataset.collater(samples)
            src_tokens = batch["net_input"]["src_tokens"]

            all_logits = musicbert.predict(  # type:ignore
                head="sequence_multitarget_tagging_head",
                tokens=src_tokens,
                return_logits=True,
            )

            for logits, target_name in zip(all_logits, target_names):
                # logits: batch x seq x vocab

                # Enumerate over batch dimension
                for logit_i, example in enumerate(logits, start=i):
                    # Trim start and end tokens:
                    data = example.detach().cpu().numpy()[1:-1]

                    out_hdfs[target_name].create_dataset(f"logits_{logit_i}", data=data)

                preds = logits.argmax(dim=-1)
                target_lengths = (
                    batch["net_input"]["src_lengths"] // args.compound_token_ratio
                )
                for line, n_tokens in zip(preds, target_lengths):
                    pred_tokens = label_dictionaries[target_name].string(
                        line[:n_tokens]
                    )
                    outfs[target_name].write(pred_tokens)
                    outfs[target_name].write("\n")

    finally:
        for outf in outfs.values():
            outf.close()

        for outf in out_hdfs.values():
            outf.close()

    shutil.copy(
        os.path.join(raw_data_dir, f"metadata_{args.dataset}.txt"),
        os.path.join(output_folder, f"metadata_{args.dataset}.txt"),
    )


if __name__ == "__main__":
    main()
