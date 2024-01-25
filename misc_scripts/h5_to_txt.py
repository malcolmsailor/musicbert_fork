import argparse

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("h5_path")
parser.add_argument("dictionary_path")
parser.add_argument("output_path")
args = parser.parse_args()

h5file = h5py.File(args.h5_path, mode="r")

logits = [h5file[x] for x in h5file]
breakpoint()
