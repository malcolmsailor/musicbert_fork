from typing import Counter

DICT = "/Users/malcolm/tmp/chord_seqs1/dict.input.txt"

vocab = set()
with open(DICT) as inf:
    for line in inf:
        token, _ = line.strip().split()
        vocab.add(token)

INPUT = "/Users/malcolm/tmp/chord_seqs1/midi_train.txt"

unknown_tokens = Counter()


# (Malcolm 2023-09-12) <s> and </s> tokens are included by fairseq automatically
with open(INPUT) as inf:
    for line in inf:
        for token in line.strip().split():
            if token not in vocab:
                unknown_tokens[token] += 1

for token, count in unknown_tokens.items():
    print(token, count)
