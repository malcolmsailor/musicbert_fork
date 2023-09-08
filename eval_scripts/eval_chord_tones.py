import argparse
import os
import sys

from fairseq.models.roberta import RobertaModel

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
PARENT_DIR = os.path.join(SCRIPT_DIR, "..")

# sys.path.append(PARENT_DIR)

# from musicbert._musicbert import MusicBERTModel

USER_DIR = os.path.join(SCRIPT_DIR, "..", "musicbert")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    return args


def main(data_dir, checkpoint):
    # musicbert = MusicBERTModel.from_pretrained(
    musicbert = RobertaModel.from_pretrained(
        # TODO: (Malcolm 2023-09-08) not sure about model_name_or_path arg
        model_name_or_path=PARENT_DIR,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir,
        user_dir=USER_DIR,
    )

    # TODO: (Malcolm 2023-09-08) musicbert.cuda()
    musicbert.eval()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.checkpoint)
