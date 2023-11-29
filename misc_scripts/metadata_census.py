import argparse
import os
from collections import Counter

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    counters = []
    splits = ("test",)
    for split in splits:
        counter = Counter()
        csv_path = os.path.join(args.input_dir, f"metadata_{split}.txt")
        df = pd.read_csv(csv_path)
        counter.update(os.path.basename(os.path.dirname(p)) for p in df.csv_path)
        counters.append(dict(counter))

    df = pd.DataFrame(counters).transpose()
    df = df.fillna(value=0)
    df.columns = [f"Num {split} examples" for split in splits]
    for col in df.columns:
        df[col] = df[col].astype(int)
    print(df)


if __name__ == "__main__":
    main()
