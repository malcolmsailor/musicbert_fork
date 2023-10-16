import argparse
from typing import Counter

parser = argparse.ArgumentParser()
parser.add_argument("dict")
parser.add_argument("input")
args = parser.parse_args()

dict = args.dict

vocab = set()
with open(dict) as inf:
    for line in inf:
        token, _ = line.strip().split()
        vocab.add(token)

input_ = args.input

unknown_tokens = Counter()


# (Malcolm 2023-09-12) <s> and </s> tokens are included by fairseq automatically
with open(input_) as inf:
    for line in inf:
        for token in line.strip().split():
            if token not in vocab:
                unknown_tokens[token] += 1

for token, count in unknown_tokens.items():
    print(token, count)
