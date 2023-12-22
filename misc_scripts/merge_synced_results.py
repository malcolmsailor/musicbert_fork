import argparse
import os
import sys
from itertools import cycle, repeat

import pandas as pd

BASE_DIR = "/Users/malcolm/output/musicbert_collated_predictions"

COLS_TO_DROP = ["balanced_accuracy"]


def process_directories(ids, csv_path, latex_path, column_prefixes):
    merged_df = pd.DataFrame()

    if column_prefixes is None:
        column_prefixes = ids

    for prefix, directory_id in zip(column_prefixes, ids):
        directory = os.path.join(BASE_DIR, directory_id)
        csv_file = os.path.join(directory, "test_metrics.csv")

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col=0)
            df = df.drop(COLS_TO_DROP, axis=1)

            df.columns = [f"{prefix} {col}" for col in df.columns]
            merged_df = pd.concat([merged_df, df], axis=1)
        else:
            print(f"File not found: {csv_file}")

    if csv_path is not None:
        merged_df.to_csv(csv_path)
        print(f"CSV output saved to {csv_path}")

    if latex_path is not None:
        to_latex_df = merged_df.loc[["KEY", "DEGREE", "QUALITY", "INVERSION"]]
        to_latex_df.index = [s.capitalize() for s in to_latex_df.index]  # type:ignore
        to_latex_df.to_latex(latex_path)
        print(f"Latex output saved to {latex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("slurm_ids", nargs="+")
    parser.add_argument("--csv-output-path", "-c")
    parser.add_argument("--latex-output-path", "-l")
    parser.add_argument("--column-prefixes", nargs="+")

    args = parser.parse_args()
    if args.column_prefixes is not None:
        assert len(args.column_prefixes) == len(args.slurm_ids)
    assert args.csv_output_path is not None or args.latex_output_path is not None

    process_directories(
        args.slurm_ids,
        args.csv_output_path,
        args.latex_output_path,
        args.column_prefixes,
    )
