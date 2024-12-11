import argparse
import json
import logging
import os
import shutil
import sys
from zipfile import ZipFile

import h5py
import torch
from fairseq.models.roberta import RobertaModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

sys.path.append(PARENT_DIR)

# from musicbert._musicbert import MusicBERTModel

USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")

LOG_INTERVAL = 50

# TODO: (Malcolm 2024-01-05) implement getting target names and dictionaries from
#   labeled dataset (which we provide path to)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--raw-dir",
        help="Required if {data_dir%bin}_raw doesn't exist, in which case this can point to the abstract_raw directory. The only files we need are the metadata files like metadata_test.txt",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="test", choices=("test", "valid", "train"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--compound-token-ratio", type=int, default=8)
    parser.add_argument("--label-dictionary-path")
    parser.add_argument("--msdebug", action="store_true")
    parser.add_argument("--overwrite", "-o", action="store_true")
    parser.add_argument("--task", default="musicbert_sequence_tagging", type=str)
    parser.add_argument(
        "--target-names",
        help="Path to target names JSON file, otherwise we look in --data-dir "
        "for a file called 'target_names.json'",
        default=None,
    )
    parser.add_argument("--head", default="sequence_tagging_head", type=str)
    # Parse once just to check for a specific unknown arg
    _, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith("--ref-dir"):
            raise ValueError(
                "Use --target-names and --label-dictionary-path instead of --ref-dir"
            )
    # Parse again so we get argparse's usual behavior on other unknown args
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir, checkpoint, output_folder_base = (
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

    output_folder = os.path.join(output_folder_base, args.dataset)

    if os.path.exists(output_folder):
        if args.overwrite:
            shutil.rmtree(output_folder)
        else:
            raise ValueError(f"Output folder {output_folder} already exists")

    assert data_dir.rstrip(os.path.sep).endswith("_bin")

    if args.target_names is not None:
        target_name_json_path = args.target_names
    else:
        target_name_json_path = os.path.join(data_dir, "target_names.json")
    if os.path.exists(target_name_json_path):
        with open(target_name_json_path, "r") as inf:
            target_names = json.load(inf)
            assert len(target_names) == 1
            target_name = target_names[0]
    else:
        target_name = "label"

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir,
        user_dir=USER_DIR,
        task="musicbert_sequence_tagging",
        label_dictionary_path=args.label_dictionary_path,
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

    # os.makedirs(os.path.join(output_folder, target_name), exist_ok=True)

    # outf = open(os.path.join(output_folder, target_name, "predictions.txt"), "w")
    outf = open(os.path.join(output_folder, "predictions", f"{target_name}.txt"), "w")
    out_hdf = h5py.File(
        os.path.join(output_folder, "predictions", f"{target_name}.h5"), "w"
    )
    label_dictionary = musicbert.task.label_dictionary
    label_dictionary.save(os.path.join(output_folder, f"{target_name}_dictionary.txt"))

    try:
        for batch_i, i in enumerate(range(0, n_examples, args.batch_size)):
            samples = [
                dataset[j] for j in range(i, min(n_examples, i + args.batch_size))
            ]
            batch = dataset.collater(samples)
            src_tokens = batch["net_input"]["src_tokens"]

            # logits: batch x seq x vocab
            logits = musicbert.predict(  # type:ignore
                head=args.head, tokens=src_tokens, return_logits=True
            )

            # Enumerate over batch dimension
            for logit_i, example in enumerate(logits, start=i):
                # Trim start and end tokens:
                data = example.detach().cpu().numpy()[1:-1]

                out_hdf.create_dataset(f"logits_{logit_i}", data=data)

            preds = logits.argmax(dim=-1)
            target_lengths = (
                batch["net_input"]["src_lengths"] // args.compound_token_ratio
            )
            for line, n_tokens in zip(preds, target_lengths):
                pred_tokens = label_dictionary.string(line[:n_tokens])
                outf.write(pred_tokens)
                outf.write("\n")
            if batch_i and (batch_i % LOG_INTERVAL == 0):
                LOGGER.info(f"Batch {batch_i}")

    finally:
        outf.close()

    metadata_basename = f"metadata_{args.dataset}.txt"
    metadata_path = os.path.join(data_dir, metadata_basename)
    zipped_data = False
    if not os.path.exists(metadata_path):
        if args.raw_dir is None:
            raw_data_dir = data_dir.rstrip(os.path.sep)[:-4] + "_raw"
            try:
                assert os.path.exists(raw_data_dir)
            except AssertionError:
                zip_data = raw_data_dir + ".zip"
                assert os.path.exists(
                    zip_data
                ), f"neither {raw_data_dir} or {zip_data} exists"
                zipped_data = True
        else:
            raw_data_dir = args.raw_dir
            if raw_data_dir.endswith(".zip"):
                zipped_data = True
                zip_data = raw_data_dir
            assert os.path.exists(
                raw_data_dir
            ), f"raw_data_dir {raw_data_dir} does not exist"

        if not zipped_data:
            metadata_path = os.path.join(raw_data_dir, metadata_basename)
        else:
            zip_file = ZipFile(zip_data)
            zip_path = os.path.join(os.path.basename(raw_data_dir), metadata_basename)
            assert zip_path in zip_file.namelist(), f"{zip_path} not in {zip_data}"
            with zip_file.open(zip_path) as src_file:
                dst = os.path.join(output_folder, metadata_basename)
                with open(dst, "wb") as dst_file:
                    dst_file.write(src_file.read())
                print(f"Copied {zip_data}::{zip_path} -> {dst}")

    shutil.copy(metadata_path, os.path.join(output_folder, metadata_basename))


if __name__ == "__main__":
    main()
