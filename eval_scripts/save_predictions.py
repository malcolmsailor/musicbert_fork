import argparse
import os
import sys

import h5py
import torch
from fairseq.models.roberta import RobertaModel

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

sys.path.append(PARENT_DIR)

# from musicbert._musicbert import MusicBERTModel

USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
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
    data_dir, checkpoint = args.data_dir, args.checkpoint
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

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir,
        user_dir=USER_DIR,
        task="musicbert_sequence_tagging",
    )

    musicbert.task.load_dataset(args.dataset)
    dataset = musicbert.task.datasets[args.dataset]

    if torch.cuda.is_available():
        musicbert.cuda()

    musicbert.eval()

    n_examples = len(dataset)
    if args.max_examples is not None:
        n_examples = min(args.max_examples, n_examples)

    os.makedirs(os.path.join(args.output_folder, "predictions"), exist_ok=True)

    outf = open(os.path.join(args.output_folder, "predictions", "predictions.txt"), "w")
    out_hdf = h5py.File(
        os.path.join(args.output_folder, "predictions", "predictions.h5"), "w"
    )
    label_dictionary = musicbert.task.label_dictionary

    try:
        for i in range(0, n_examples, args.batch_size):
            samples = [
                dataset[j] for j in range(i, min(n_examples, i + args.batch_size))
            ]
            batch = dataset.collater(samples)
            src_tokens = batch["net_input"]["src_tokens"]

            # logits: batch x seq x vocab
            logits = musicbert.predict(  # type:ignore
                head="sequence_tagging_head", tokens=src_tokens, return_logits=True
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

    finally:
        outf.close()


if __name__ == "__main__":
    main()
