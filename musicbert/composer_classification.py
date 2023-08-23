import logging
import os

import torch
from fairseq.data import Dictionary, LanguagePairDataset, data_utils
from fairseq.tasks import LegacyFairseqTask, register_task

from musicbert._musicbert import OctupleTokenDataset

LOGGER = logging.getLogger(__name__)


@register_task("composer_classification")
class ComposerClassificationTask(LegacyFairseqTask):
    def __init__(self, args, input_vocab, label_vocab):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.targets_vocab = label_vocab

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        input_vocab = Dictionary.load(os.path.join(args.data, "dict.input.txt"))
        label_vocab = Dictionary.load(os.path.join(args.data, "dict.targets.txt"))
        print("| [input] dictionary: {} types".format(len(input_vocab)))
        print("| [label] dictionary: {} types".format(len(label_vocab)))

        return ComposerClassificationTask(args, input_vocab, label_vocab)  # type:ignore

    def _load_octuple_data(self, data_path):
        input0 = data_utils.load_indexed_dataset(
            data_path,
            self.input_vocab,
            # TODO: (Malcolm 2023-08-23) not sure what `dataset_impl` does
            self.args.dataset_impl,  # type:ignore
        )
        src_dataset = OctupleTokenDataset(input0)
        return src_dataset

    def _load_targets_data(self, targets_data_path):
        labels = []

        with open(targets_data_path) as file:
            for line in file:
                label = line.strip()
                labels.append(
                    # Convert label to a numeric ID.
                    torch.LongTensor([self.targets_vocab.add_symbol(label)])
                )
        return labels

    def load_dataset(self, split, **kwargs):
        input_data_path = os.path.join(self.args.data, "input0", split)
        src_dataset = self._load_octuple_data(input_data_path)
        targets_data_path = os.path.join(self.args.data, "label", split)
        labels = self._load_targets_data(targets_data_path)
        assert len(labels) == len(src_dataset)
        LOGGER.info(("| {} {} {} examples".format(self.args.data, split, len(labels))))
        self.datasets[split] = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.input_vocab,
            tgt=labels,
            tgt_sizes=torch.ones(len(labels)),
            tgt_dict=self.targets_vocab,
            left_pad_source=False,
            input_feeding=False,
        )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_positions, 1)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.targets_vocab
