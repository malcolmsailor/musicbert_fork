"""
Example testing command:
python eval_scripts/save_multi_task_predictions.py --data-dir ~/project/datasets/chord_tones/fairseq/many_target_bin --checkpoint ~/project/new_checkpoints/musicbert_fork/32702693/checkpoint_best.pt --output-folder ~/tmp/mout --msdebug --ignore-specials 4 --overwrite --max-examples 2
python eval_scripts/save_multi_task_predictions.py \
    --data-dir ~/output/test_data/chord_tones_bin \
    --checkpoint /Volumes/Zarebski/musicbert_checkpoints/32702693/checkpoint_best.pt \
    --output-folder ~/tmp/mout --msdebug --ignore-specials 4 \
    --overwrite --max-examples 2
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")
sys.path.append(PARENT_DIR)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


LOG_INTERVAL = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="assumed to end in '_bin' and have an equivalent ending in '_raw' that contains 'metadata_test.txt'",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="a directory that contains `target_names.json` as well as "
        "`label[x]/dict.txt` files. If not provided, the value of "
        "--data-dir is used.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="test", choices=("test", "valid", "train"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--compound-token-ratio", type=int, default=8)
    parser.add_argument("--msdebug", action="store_true")
    parser.add_argument("--overwrite", "-o", action="store_true")
    parser.add_argument("--ignore-specials", type=int, default=4)
    parser.add_argument(
        "--task", default="musicbert_multitask_sequence_tagging", type=str
    )
    parser.add_argument("--head", default="sequence_multitask_tagging_head", type=str)

    args = parser.parse_args()
    return args


def extract_features_with_conditioning(
    self,
    tokens: torch.LongTensor,
    z_tokens: torch.LongTensor,
    return_all_hiddens: bool = False,
) -> torch.Tensor:
    # Due to fairseq's somewhat weird import system overriding RobertaHubInterface
    #   is somewhat tricky, so instead we monkey-patch
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)  # type:ignore
    if tokens.size(-1) > self.model.max_positions():
        raise ValueError(
            "tokens exceeds maximum length: {} > {}".format(
                tokens.size(-1), self.model.max_positions()
            )
        )
    if z_tokens.dim() == 1:
        z_tokens.unsqueeze(0)

    features, extra = self.model(
        tokens.to(device=self.device),  # type:ignore
        features_only=True,
        return_all_hiddens=return_all_hiddens,
        z_tokens=z_tokens.to(device=self.device),  # type:ignore
    )
    if return_all_hiddens:
        # convert from T x B x C -> B x T x C
        inner_states = extra["inner_states"]
        return [
            inner_state.transpose(0, 1) for inner_state in inner_states
        ]  # type:ignore
    else:
        return features  # just the last layer's features


def predict_with_conditioning(
    self,
    head: str,
    tokens: torch.LongTensor,
    z_tokens: torch.LongTensor,
    return_logits: bool = False,
):
    features = self.extract_features(
        tokens.to(device=self.device), z_tokens=z_tokens.to(device=self.device)
    )  # type:ignore
    backward_compatible_head = head.replace("multitask", "multitarget")
    
    if head not in self.model.classification_heads and backward_compatible_head in self.model.classification_heads:
        head = backward_compatible_head

    logits = self.model.classification_heads[head](features)
    if return_logits:
        return logits
    return F.log_softmax(logits, dim=-1)


def main():
    args = parse_args()

    data_dir, ref_dir, checkpoint, output_folder_base = (
        args.data_dir,
        args.ref_dir,
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

    if ref_dir is None:
        ref_dir = data_dir

    output_folder = os.path.join(output_folder_base, args.dataset)

    if os.path.exists(output_folder):
        if args.overwrite:
            shutil.rmtree(output_folder)
        else:
            raise ValueError(f"Output folder {output_folder} already exists")

    assert data_dir.rstrip(os.path.sep).endswith("_bin")

    with open(os.path.join(ref_dir, "target_names.json"), "r") as inf:
        target_names = json.load(inf)

    if args.task == "musicbert_conditioned_multitask_sequence_tagging":
        RobertaHubInterface.extract_features = (
            extract_features_with_conditioning
        )  # type:ignore
        RobertaHubInterface.predict = predict_with_conditioning  # type:ignore

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir,
        user_dir=USER_DIR,
        task=args.task,
        ref_dir=args.ref_dir,
        target_names=target_names,
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
        for batch_i, i in enumerate(range(0, n_examples, args.batch_size)):
            samples = [
                dataset[j] for j in range(i, min(n_examples, i + args.batch_size))
            ]
            batch = dataset.collater(samples)
            src_tokens = batch["net_input"]["src_tokens"]

            predict_kwargs = {
                # TODO rename
                "head": args.head,
                "tokens": src_tokens,
                "return_logits": True,
            }

            if "z_tokens" in batch:
                predict_kwargs["z_tokens"] = batch["z_tokens"]

            all_logits = musicbert.predict(  # type:ignore
                **predict_kwargs
            )

            for logits, target_name in zip(all_logits, target_names):
                # logits: batch x seq x vocab

                # Enumerate over batch dimension
                for logit_i, example in enumerate(logits, start=i):
                    # Trim start and end tokens:
                    data = example.detach().cpu().numpy()[1:-1]

                    # Don't save specials
                    if args.ignore_specials:
                        data = data[:, args.ignore_specials :]

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
            if batch_i and (batch_i % LOG_INTERVAL == 0):
                LOGGER.info(f"Batch {batch_i}")
    finally:
        for outf in outfs.values():
            outf.close()

        for outf in out_hdfs.values():
            outf.close()

    metadata_basename = f"metadata_{args.dataset}.txt"
    metadata_path = os.path.join(data_dir, metadata_basename)
    if not os.path.exists(metadata_path):
        raw_data_dir = data_dir.rstrip(os.path.sep)[:-4] + "_raw"
        assert os.path.exists(raw_data_dir)
        metadata_path = os.path.join(raw_data_dir, metadata_basename)

    shutil.copy(metadata_path, os.path.join(output_folder, metadata_basename))

    if args.ignore_specials:
        with open(os.path.join(output_folder, "num_ignored_specials.txt"), "w") as outf:
            outf.write(str(args.ignore_specials))


if __name__ == "__main__":
    main()
