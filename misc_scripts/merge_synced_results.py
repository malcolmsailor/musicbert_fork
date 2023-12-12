import os
import sys

import pandas as pd

BASE_DIR = "/Users/malcolm/output/musicbert_collated_predictions"

COLS_TO_DROP = ["balanced_accuracy"]


def process_directories(ids, output_path):
    merged_df = pd.DataFrame()

    for directory_id in ids:
        directory = os.path.join(BASE_DIR, directory_id)
        csv_file = os.path.join(directory, "test_metrics.csv")

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col=0)
            df = df.drop(COLS_TO_DROP, axis=1)
            # Append directory_id to each column except for the index column
            df.columns = [
                f"{directory_id}_{col}" if col != df.index.name else col
                for col in df.columns
            ]
            merged_df = pd.concat([merged_df, df], axis=1)
        else:
            print(f"File not found: {csv_file}")

    merged_df.to_csv(output_path)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py [id1] [id2] ... [output_path]")
        sys.exit(1)

    ids = sys.argv[1:-1]
    output_path = sys.argv[-1]

    process_directories(ids, output_path)
