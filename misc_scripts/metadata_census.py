import argparse
import os
from collections import Counter, defaultdict
from itertools import chain

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    splits = (
        "train",
        "valid",
        "test",
    )
    dict_accumulator = []
    col_names = []
    for split in splits:
        example_counter = Counter()
        unique_scores = defaultdict(set)
        csv_path = os.path.join(args.input_dir, f"metadata_{split}.txt")
        if not os.path.exists(csv_path):
            print(f"Warning, {csv_path} does not exist")
            continue
        df = pd.read_csv(csv_path)
        for p in df.csv_path:
            dir_name, score_name = os.path.split(p)
            corpus_name = os.path.basename(dir_name)
            example_counter[corpus_name] += 1
            unique_scores[corpus_name].add(score_name)

        dict_accumulator.append(dict(example_counter))
        dict_accumulator.append(
            {corpus_name: len(scores) for corpus_name, scores in unique_scores.items()}
        )
        col_names.append(split)

    df = pd.DataFrame(dict_accumulator).transpose()
    df = df.fillna(value=0)
    example_col_names = [f"Num {split} examples" for split in col_names]
    score_col_names = [f"Num {split} unique scores" for split in col_names]
    df.columns = list(chain(*zip(example_col_names, score_col_names)))
    for col in df.columns:
        df[col] = df[col].astype(int)
    df.loc["total"] = df.sum()
    print(df)


if __name__ == "__main__":
    main()
