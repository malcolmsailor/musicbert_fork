import argparse

import pandas as pd
import sklearn.metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)
    breakpoint()


if __name__ == "__main__":
    main()
