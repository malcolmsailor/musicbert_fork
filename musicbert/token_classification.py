"""Code in this file is based on this (closed and not merged) PR:
https://github.com/facebookresearch/fairseq/pull/1709/files
"""


import logging
import math
import os
import warnings
from typing import Literal, Sequence

import numpy as np
import sklearn.metrics  # type:ignore
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import (
    Dictionary,
    FairseqDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    ReplaceDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
)
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.model import RobertaModel
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.tasks import FairseqTask, register_task
from torch import nn

LOGGER = logging.getLogger(__name__)

PAD_IDX = 1

# TODO: (Malcolm 2023-09-15) remove
TEMP_CACHE = {}


class AssertSameLengthDataset(FairseqDataset):
    def __init__(self, first, second, first_to_second_ratio: int = 1):
        self.first = first
        self.second = second
        self.first_to_second_ratio = first_to_second_ratio

    def __getitem__(self, index):
        assert (
            torch.numel(self.first[index])
            == torch.numel(self.second[index]) * self.first_to_second_ratio
        )

    def __len__(self):
        return 0

    def collater(self, samples):
        return 0


class RobertaSequenceTaggingHead(nn.Module):
    """Head for sequence tagging/token-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. "
                    "This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        # TODO: (Malcolm 2023-09-05)
        # https://github.com/facebookresearch/fairseq/pull/1709/files#r381391530
        # Would it make sense to add layer_norm here just like in the RobertaLMHead?
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def get_loss_weights(dictionary, weight_coef):
    counts = []
    n_appended_tokens = 0
    for i in range(dictionary.nspecial, len(dictionary)):
        word = dictionary.symbols[i]
        if word.startswith("madeupword"):
            n_appended_tokens = len(dictionary) - i
            break
        counts.append(dictionary.count[i])
    counts = torch.tensor(counts)
    proportions = counts / counts.sum()
    inv_proportions = 1 - proportions
    adjusted = len(counts) * inv_proportions
    interpolated = (1 - weight_coef) + weight_coef * adjusted
    # It looks like we don't want to append the interpolated tokens since the targets
    #   only have the specials plus the actual tokens
    out = torch.concat([torch.ones(dictionary.nspecial), interpolated])
    return out


@register_criterion("sequence_tagging")
class SequenceTaggingCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.pad_idx = task.label_dictionary.pad()
        self.compound_token_ratio = self.task.args.compound_token_ratio
        if self.task.args.weight_loss:
            self.loss_weights = get_loss_weights(
                task.label_dictionary, self.task.args.weight_loss_coef
            )
            self.set_loss_weight_device = False
        else:
            self.loss_weights = None
            self.set_loss_weight_device = True

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--classification-head-name",
            default="sequence_tagging_head",
            help="name of the classification head to use",
        )
        parser.add_argument("--compound-token-ratio", type=int, default=1)
        parser.add_argument("--weight-loss", action="store_true", default=False)
        parser.add_argument(
            "--weight-loss-coef",
            type=float,
            default=0.5,
            help="scales loss weights where 0=no weighting and 1=weighting by token frequency",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sequence_tagging"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )

        targets = model.get_targets(sample, [logits]).view(-1)
        adjusted_ntokens = sample["ntokens"] // self.compound_token_ratio
        nsentences = sample["target"].size(0)
        sample_size = adjusted_ntokens - nsentences  # number of tokens without eos

        if not self.set_loss_weight_device:
            # Hack to get the device of the model
            device = next(model.parameters()).device

            assert self.loss_weights is not None
            self.loss_weights = self.loss_weights.to(device)
            self.set_loss_weight_device = True

        logits = logits.view(-1, logits.size(-1))
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            ignore_index=self.pad_idx,
            reduction="sum",
            # weight=self.loss_weights,
        )

        # To get the same behavior as the original implementation we should ignore
        #   all specials, not just pad. Not sure if we want to do this.

        masked_preds = logits[targets != self.pad_idx].argmax(dim=1)
        masked_targets = targets[targets != self.pad_idx]
        logging_output = {
            "loss": loss.data,
            "ntokens": adjusted_ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "ncorrect": utils.item((masked_preds == masked_targets).sum()),
            "y_true": masked_targets.detach().cpu().numpy(),
            "y_pred": masked_preds.detach().cpu().numpy(),
            # because `reduce_metrics` is a static method we need to
            #   include the following in the logging output. Although I wonder
            #   what would happen if we just removed `@staticmethod`
            "nspecial": self.task.label_dictionary.nspecial,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        # TODO: (Malcolm 2023-09-11) A few things I don't understand here:
        #   1. why divide loss by log of 2?
        #   2. why is "loss" divided by sample_size and "nll_loss"  is divided by
        #       ntokens?
        #           Note that ntokens should be the number of tokens including <eos>
        #               and sample_size should be the number of tokens excluding <eos>
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # sample_size should be the number of tokens w/o the
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "y_pred" in logging_outputs[0]:
            y_pred = np.concatenate(
                tuple(log.get("y_pred") for log in logging_outputs if "y_pred" in log)
            )
            y_true = np.concatenate(
                tuple(log.get("y_true") for log in logging_outputs if "y_true" in log)
            )
            for average in ["micro", "weighted"]:
                (
                    precision,
                    recall,
                    f1,
                    support,
                ) = sklearn.metrics.precision_recall_fscore_support(
                    y_true, y_pred, average=average, zero_division=0.0  # type:ignore
                )
                metrics.log_scalar(f"precision_{average}", precision)  # type:ignore
                metrics.log_scalar(f"recall_{average}", recall)  # type:ignore
                metrics.log_scalar(f"f1_{average}", f1)  # type:ignore

            balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
            metrics.log_scalar(f"balanced_accuracy", balanced_accuracy)

            n_special = 4
            no_specials_mask = y_true >= n_special  # type:ignore
            confused = sklearn.metrics.confusion_matrix(
                y_true[no_specials_mask] - n_special,
                y_pred[no_specials_mask] - n_special,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                warnings.filterwarnings(
                    "ignore",
                    message="y_pred contains classes not in y_true",
                    category=UserWarning,
                )
                precision_per_class = np.nan_to_num(
                    confused.diagonal() / confused.sum(axis=0)
                )
                recall_per_class = np.nan_to_num(
                    confused.diagonal() / confused.sum(axis=1)
                )
                f1_per_class = np.nan_to_num(
                    (2 * precision_per_class * recall_per_class)
                    / (precision_per_class + recall_per_class)
                )

            # TODO: (Malcolm 2023-09-12) At some point don't hardcode labels
            labels = ["yes", "no"]

            for (
                label_i,
                label,
            ) in enumerate(labels):
                # specials may or may not be included in the metric arrays, but we
                #   don't want to log them. So instead we do as follows:
                class_i = len(precision_per_class) - len(labels) + label_i
                metrics.log_scalar(f"precision_{label}", precision_per_class[class_i])
                metrics.log_scalar(f"recall_{label}", recall_per_class[class_i])
                metrics.log_scalar(f"f1_{label}", f1_per_class[class_i])

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


# class MusicBERTSequenceTaggingModel(RobertaModel):
#     def register_sequence_tagging_head(
#         self, name, num_classes=None, inner_dim=None, **kwargs
#     ):
#         """Register a classification head."""
#         if name in self.classification_heads:
#             prev_num_classes = self.classification_heads[  # type:ignore
#                 name
#             ].out_proj.out_features  # type:ignore
#             prev_inner_dim = self.classification_heads[  # type:ignore
#                 name
#             ].dense.out_features  # type:ignore
#             if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
#                 LOGGER.warning(
#                     're-registering head "{}" with num_classes {} (prev: {}) '
#                     "and inner_dim {} (prev: {})".format(
#                         name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
#                     )
#                 )
#         self.classification_heads[name] = RobertaSequenceTaggingHead(  # type:ignore
#             input_dim=self.args.encoder_embed_dim,  # type:ignore
#             inner_dim=inner_dim or self.args.encoder_embed_dim,  # type:ignore
#             num_classes=num_classes,
#             activation_fn=self.args.pooler_activation_fn,  # type:ignore
#             pooler_dropout=self.args.pooler_dropout,  # type:ignore
#             q_noise=self.args.quant_noise_pq,  # type:ignore
#             qn_block_size=self.args.quant_noise_pq_block_size,  # type:ignore
#             do_spectral_norm=self.args.spectral_norm_classification_head,  # type:ignore
#         )


@register_task("musicbert_sequence_tagging")
class SequenceTaggingTask(FairseqTask):
    """
    Sequence tagging (also called sentence tagging or sequence labelling) task that predicts a class for each input token.
    Inputs should be stored in 'input0' directory, labels in 'label' directory.
    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    upsample_encoder: bool = False

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        # -1 raises an exception below; --num-classes is required
        parser.add_argument(
            "--num-classes", type=int, default=-1, help="number of classes (required)"
        )
        parser.add_argument("--msdebug", action="store_true")
        # (Malcolm 2023-09-05) not sure why we would want to not shuffle
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument("--freeze-layers", type=int, default=-1)

    def __init__(self, args, data_dictionary, label_dictionary):
        if args.msdebug:
            import pdb
            import sys
            import traceback

            def custom_excepthook(exc_type, exc_value, exc_traceback):
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout
                )
                pdb.post_mortem(exc_traceback)

            sys.excepthook = custom_excepthook
        super().__init__(args)
        self.dictionary = data_dictionary
        # (Malcolm 2023-09-12) for printing label names to work above these
        #   assertions need to be correct. If we remove the staticmethod decorator
        #   we could probably get rid of this
        assert label_dictionary[4] == "yes"
        assert label_dictionary[5] == "no"
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            # TODO: (Malcolm 2023-09-08) this will raise an attribute error
            # We just provide max positions as an arg
            raise NotImplementedError("Provide '--max-positions'")
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions  # tuple[int, int] ?
        # The code from the PR seems to assume that the task has an `args attribute`
        self.args = args

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        # (Malcolm 2023-09-05) We need the <mask> symbol not because we use it but
        #   so that the dictionary sizes match with the pretrained checkpoints.
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        LOGGER.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        label_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "label", "dict.txt"),
            source=False,
        )
        LOGGER.info("[label] dictionary: {} types".format(len(label_dict)))
        return SequenceTaggingTask(args, data_dict, label_dict)  # type:ignore

    def print_examples(
        self,
        epoch,
        model: nn.Module,
        split: Literal["train", "valid"],
        indices: Sequence[int],
        max_tokens_to_print=16,
        token_length=1,
    ):
        model_state = model.training
        model.eval()
        dataset = self.datasets[split]
        samples = [dataset[i] for i in indices]
        batch = dataset.collater(samples)

        # Hack to get the device of the model
        device = next(model.parameters()).device

        # Move input to model device
        net_input = {k: v.to(device) for k, v in batch["net_input"].items()}

        logits, _ = model(
            **net_input,
            features_only=True,
            classification_head_name="sequence_tagging_head",
        )
        logits = logits.to("cpu")
        preds = logits.argmax(dim=-1)
        total_correct = 0
        total = 0
        for i, (pred, target) in enumerate(zip(preds, batch["target"])):
            # Ignore padding/bos/eos
            valid_mask = target >= 0
            pred = pred[valid_mask]
            target = target[valid_mask]
            total += valid_mask.sum()
            total_correct += (pred == target).sum()

            #
            pred = pred[:max_tokens_to_print]
            target = target[:max_tokens_to_print]

            # We need to adjust for the specials at the beginning
            #   of the dictionary
            # pred += self.label_dictionary.nspecial
            # target += self.label_dictionary.nspecial

            target_tokens = self.label_dictionary.string(target)
            pred_tokens = self.label_dictionary.string(pred)
            target_tokens = [
                f"{x[:token_length]:<{token_length}}" for x in target_tokens.split()
            ]
            pred_tokens = [
                f"{x[:token_length]:<{token_length}}" for x in pred_tokens.split()
            ]
            target_tokens = " ".join(target_tokens)
            pred_tokens = " ".join(pred_tokens)

            LOGGER.info(f"Epoch {epoch} {split} target     {i + 1}: {target_tokens}")
            LOGGER.info(f"Epoch {epoch} {split} prediction {i + 1}: {pred_tokens}")

        model.train(model_state)

    # (Malcolm 2023-09-18) Leaving this for now, used it to debug freezing issue
    # def begin_epoch(self, epoch, model):
    #     for name, param in model.named_parameters():
    #         ex_weight = param.data.detach().reshape(-1)[0].item()
    #         if name in TEMP_CACHE:
    #             prev_weight = TEMP_CACHE[name]
    #             equals = ex_weight == prev_weight
    #             print(
    #                 f"{name}: {'equal    ' if equals else 'not equal'} {prev_weight} {ex_weight}"
    #             )
    #         TEMP_CACHE[name] = ex_weight

    #     l1_weight = (
    #         model.encoder.sentence_encoder.layers[0]
    #         .fc1.weight.data.reshape(-1)[0]
    #         .item()
    #     )
    #     l2_weight = (
    #         model.encoder.sentence_encoder.layers[1]
    #         .fc1.weight.data.reshape(-1)[0]
    #         .item()
    #     )
    #     c_weight = (
    #         model.classification_heads.sequence_tagging_head.dense.weight.data.reshape(
    #             -1
    #         )[0].item()
    #     )
    #     # if model.encoder.sentence_encoder.layers[1].fc1.weight.requires_grad:
    #     #     print(l1_weight)
    #     #     print(l2_weight)
    #     #     print(c_weight)
    #     #     breakpoint()

    def begin_valid_epoch(self, epoch, model):
        """As a sanity check, print out example outputs for training and validation sets."""

        self.print_examples(epoch, model, "train", [0, 1, 2, 3])
        self.print_examples(epoch, model, "valid", [0, 1, 2, 3])

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)  # type:ignore

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,  # type:ignore
                combine=combine,
            )
            assert dataset is not None, "could not find dataset: {}".format(
                get_path(type, split)
            )
            return dataset

        src_tokens = make_dataset("input0", self.source_dictionary)

        label_dataset = make_dataset("label", self.label_dictionary)

        # (Malcolm 2023-09-08) The code that I based this off of includes the
        #   following commented out lines so that we only predict items in the
        #   target vocabulary and not specials. However that doesn't seem necessary
        #   and there seem to be some weird bugs going on so I'm disabling that for
        #   now.

        # OffsetTokensDataset offsets tokens to get the targets to the
        # correct range (0,1,2,...)
        # label_dataset = OffsetTokensDataset(
        #     label_dataset,
        #     offset=-self.label_dictionary.nspecial,
        # )
        # ReplaceDataset replaces specials (bos, eos, and existing padding used when some
        # tokens should not be predicted) with -1
        # label_dataset = ReplaceDataset(
        #     label_dataset,
        #     replace_map={i: -1 for i in range(-self.label_dictionary.nspecial, 0)},
        #     offsets=np.zeros(len(label_dataset), dtype=int),
        # )
        # RightPadDataset uses -1 as padding, will be used to mask out padding
        # when calculating loss
        label_dataset = RightPadDataset(
            label_dataset, pad_idx=self.label_dictionary.pad()
        )
        assert self.label_dictionary.pad() == self.source_dictionary.pad() == PAD_IDX

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "target": label_dataset,
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "_assert_lengths_match": AssertSameLengthDataset(
                src_tokens, label_dataset, self.args.compound_token_ratio  # type:ignore
            ),
        }

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:  # type:ignore
            dataset = nested_dataset
        else:
            with data_utils.numpy_seed(self.args.seed):  # type:ignore
                shuffle = np.random.permutation(len(src_tokens))
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        LOGGER.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        if args.freeze_layers > 0:
            LOGGER.info(f"Freezing {args.freeze_layers=} layers")
            # What we *don't* want to freeze:
            # 1. the last n - freeze_layers encoder layers
            # 2. the classification head
            n_layers = len(model.encoder.sentence_encoder.layers)
            assert n_layers >= args.freeze_layers

            for parameter in model.parameters():
                parameter.requires_grad = False

            for layer_i, layer in enumerate(model.encoder.sentence_encoder.layers):
                if layer_i < args.freeze_layers:
                    continue
                for parameter in layer.parameters():
                    parameter.requires_grad = True

            if model.encoder.sentence_encoder.upsample:
                for parameter in model.encoder.sentence_encoder.upsampling.parameters():
                    parameter.requires_grad = True

        # We register the sequence tagging head after any freezing so that it won't
        #   be frozen
        model.register_sequence_tagging_head(
            getattr(args, "classification_head_name", "sequence_tagging_head"),
            num_classes=self.args.num_classes + self.label_dictionary.nspecial,
            sequence_tagging=True,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
