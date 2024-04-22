# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import logging
import math
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import fairseq.tasks.masked_lm
import fairseq.tasks.sentence_prediction
import numpy as np
import sklearn.metrics  # type:ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.data import (
    LanguagePairDataset,
    MaskTokensDataset,
    PrependTokenDataset,
    data_utils,
)
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import (  # RobertaEncoder,
    RobertaModel,
    TransformerSentenceEncoder,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

# TODO: (Malcolm 2023-09-18) if I'm not actually using the freezable encoder,
#   I should probably remove it. (I implemented per-layer freezing elsewhere, the
#   FreezableRobertaEncoder freezes everything except the lm head. It also
#   disables dropout which is probably good. Less obvious how to disable
#   dropout on a per-layer basis.)
from musicbert.freezable_roberta import FreezableRobertaEncoder
from musicbert.token_classification import RobertaSequenceTaggingHead
from musicbert.token_classification_multi_task import (
    RobertaSequenceMultiTaggingHead,
    RobertaSequenceConditionalMultiTaggingHead,
)

warnings.filterwarnings("ignore", message=".*NVIDIA's apex library.*")

LOGGER = logging.getLogger(__name__)

DISABLE_CP = False
if DISABLE_CP:
    raise NotImplementedError

MASK_STRATEGY = (
    os.environ["mask_strategy"].split("+") if "mask_strategy" in os.environ else ["bar"]
)
assert all(item in ["element", "compound", "bar"] for item in MASK_STRATEGY)
# LOGGER.info(f"MASK_STRATEGY = {MASK_STRATEGY}")


CONVERT_ENCODING = (
    os.environ["convert_encoding"] if "convert_encoding" in os.environ else "OCTMIDI"
)
# LOGGER.info(f"CONVERT_ENCODING = {CONVERT_ENCODING}")

CROP_LENGTH = int(os.environ["crop_length"]) if "crop_length" in os.environ else None
LOGGER.info(f"CROP_LENGTH = {CROP_LENGTH}")  # of compound tokens

MAX_BARS = 256
MAX_INSTRUMENTS = 256


# Thank GitHub user @neelansh for providing multi-label classification solution
# See https://github.com/pytorch/fairseq/issues/2169
@register_task("sentence_prediction_multilabel")
class MusicBERTSentencePredictionMultilabelTask(SentencePredictionTask):
    def load_dataset(self, split, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, "input0", split)  # type:ignore
        input0 = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,  # type:ignore
            combine=combine,
        )
        if self.args.init_token is not None:  # type:ignore
            input0 = OctupleTokenDataset(input0)
        src_dataset = input0
        labels, label_lengths = [], []
        with open(
            os.path.join(self.args.data, "label", split + ".label")  # type:ignore
        ) as file:
            for line in file:
                line = line.strip()
                line = line.split()
                label = [
                    self.label_dictionary.index(item) for item in line  # type:ignore
                ]

                if len(label) < self.args.num_classes:  # type:ignore
                    label = label + [
                        self.label_dictionary.index("<pad>")  # type:ignore
                    ] * (
                        self.args.num_classes - len(label)  # type:ignore
                    )

                label = label[: self.args.num_classes]  # type:ignore

                label = torch.tensor(label)
                labels.append(label)
                label_lengths.append(len(label))
        assert src_dataset is not None
        assert len(src_dataset) == len(labels)
        self.datasets[split] = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            # TODO: (Malcolm 2023-08-23) this looks wrong
            src_dict=self.label_dictionary,  # type:ignore
            tgt=labels,
            tgt_sizes=torch.tensor(label_lengths),
            tgt_dict=self.label_dictionary,  # type:ignore
            left_pad_source=False,
            input_feeding=False,
        )


# Thank GitHub user @neelansh for providing multi-label classification solution
# See https://github.com/pytorch/fairseq/issues/2169
@register_criterion("sentence_prediction_multilabel")
class MusicBERTSentencePredictionMultilabelCriterion(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"
        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits])
        targets = F.one_hot(targets.long(), num_classes=logits.size()[-1] + 4)
        targets = targets.sum(dim=1)
        targets = targets[:, 4:]
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="sum"
        )
        sample_size = logits.size()[0]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size * logits.size()[1],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds = F.relu(torch.sign(logits))
        logging_output["ncorrect"] = (
            sample_size - torch.sign((preds != targets).sum(dim=1)).sum().data
        )
        logging_output["y_true"] = targets.detach().cpu().numpy()
        logging_output["y_pred"] = torch.sigmoid(logits).detach().cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "y_pred" in logging_outputs[0]:
            y_pred = np.vstack(
                tuple(log.get("y_pred") for log in logging_outputs if "y_pred" in log)
            )
            y_true = np.vstack(
                tuple(log.get("y_true") for log in logging_outputs if "y_true" in log)
            )
            for score in ["roc_auc_score", "f1_score"]:
                for average in ["macro", "micro", "weighted", "samples"]:
                    try:
                        y_score = (
                            np.round(y_pred)  # type:ignore
                            if score == "f1_score"
                            else y_pred
                        )
                        kwargs = {"zero_division": 0} if score == "f1_score" else dict()
                        result = sklearn.metrics.__dict__[score](
                            y_true, y_score, average=average, **kwargs
                        )
                        metrics.log_scalar("{}_{}".format(score, average), result)
                    except BaseException as e:
                        metrics.log_scalar("{}_{}".format(score, average), float("inf"))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False


class OctupleMaskTokensDataset(MaskTokensDataset):
    # Why are we caching? That seems to make no sense given that the dataset presumably
    #   has very many items
    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)
            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )
            # TODO: (Malcolm 2023-08-15) what do they mean by this?
            assert not self.mask_whole_words, "mask whole words not supported for cp"

            def generate_mask(sz, prob):
                # Shape (sz)
                mask_n = np.random.rand(sz)

                # 0 means no mask
                mask_s = np.zeros(sz, dtype=np.int8)

                # We sum bools 3 times; thus if
                #   mask_n < p * random_token_prob, mask_s = 3
                mask_s += mask_n < prob * (self.random_token_prob)  # 3 -> random
                mask_s += mask_n < prob * (
                    self.random_token_prob + self.leave_unmasked_prob
                )  # 2 -> original
                mask_s += mask_n < prob * 1.00  # 1 -> [mask]
                return mask_s

            mask_prob = self.mask_prob
            mask = np.zeros_like(item, dtype=np.int8)
            # mask bos eos tokens (compound)
            mask[:8] = np.repeat(generate_mask(1, mask_prob), 8)
            # mask bos eos tokens (compound)
            mask[-8:] = np.repeat(generate_mask(1, mask_prob), 8)

            # TODO: (Malcolm 2023-08-15) contrary to paper, they seem to be randomly
            #   sampling a mask strategy
            strategy = np.random.choice(MASK_STRATEGY)
            if strategy == "element":  # element level mask
                mask[8:-8] = np.repeat(generate_mask(sz - 2 * 8, mask_prob), 1)
            if strategy == "compound":  # compound token level mask
                mask[8:-8] = np.repeat(generate_mask(sz // 8 - 2, mask_prob), 8)
            if strategy == "bar":  # bar level mask
                mask[8:-8] = (
                    generate_mask(
                        (MAX_BARS * MAX_INSTRUMENTS + len(self.vocab)) * 8, mask_prob
                    )
                    .reshape(-1, 8)[
                        ((item[8:-8:8] - 4) * MAX_INSTRUMENTS)
                        + (item[8 + 2 : -8 + 2 : 8] - 4)
                    ]
                    .flatten()
                )
            if self.return_masked_tokens:
                new_item = item.numpy()[:]
                new_item[mask == 0] = self.pad_idx
                return torch.from_numpy(new_item)
            masked_item = np.random.choice(len(self.vocab), sz)
            set_original = np.isin(mask, [0, 2])
            masked_item[set_original] = item[set_original]
            set_mask = np.isin(mask, [1])
            masked_item[set_mask] = self.mask_idx
            return torch.from_numpy(masked_item)


class OctupleEncoder(TransformerSentenceEncoder):
    def __init__(self, *args, upsample: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tpu = False
        embedding_dim = kwargs["embedding_dim"]
        self.downsampling = nn.Sequential(nn.Linear(embedding_dim * 8, embedding_dim))
        self.upsample = upsample
        # (Malcolm 2023-09-07) if `self.upsample` is False, then we shouldn't use
        #   self.upsampling, but we nevertheless create it so that the model will
        #   match the pretrained checkpoints. (We could hack `fairseq` so that
        #   strict=False; there doesn't seem to be a cmd line arg for that.)
        self.upsampling = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 8))

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: Optional[torch.Tensor] = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]
    ]:
        # tokens: batch * compound_seq
        assert tokens.ndim == 2

        batch, compound_seq = tokens.shape

        ratio = 1 if DISABLE_CP else 8

        seq = compound_seq / ratio

        if not DISABLE_CP:
            assert (
                tokens.shape[1] % ratio == 0
            ), f"token sequences length should be multiple of {ratio} for compound mode"
            # (Malcolm 2024-02-20) I'm not sure what the motivation for this assertion
            #   is, hidden states for intermediate layers seem to work as normal.
            # assert last_state_only, "hidden states not available for compound mode"
            assert (
                positions is None
            ), "custom positions are not supported for compound mode"
            assert (
                token_embeddings is None
            ), "custom token embeddings are not supported for compound mode"
            assert (
                segment_labels is None
            ), "segment embedding not supported for compound mode"

        # padding mask: boolean tensor (batch, seq)
        #   where seq = compound_seq // 8
        padding_mask = tokens[:, ::ratio].eq(self.padding_idx)
        assert padding_mask.shape == (batch, seq)

        if not self.traceable and not self.tpu and not padding_mask.any():
            # ?
            padding_mask = None

        if token_embeddings is not None:
            # TODO: (Malcolm 2023-08-14) ?
            x = token_embeddings

        else:
            # x: batch, compound_seq, embedding_dim
            x = self.embed_tokens(tokens)

        # assert x.shape == (batch, compound_seq, self.embedding_dim)

        if not DISABLE_CP:
            # Project from (batch, compound_seq, embedding) -> (batch, seq, ratio * embedding )

            unflattened = x.view(x.shape[0], x.shape[1] // ratio, -1)
            x = self.downsampling(unflattened)

        # assert x.shape == (batch, seq, ratio * self.embedding_dim)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        # Add positional embeddings
        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens[:, ::ratio], positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)

        # TODO: (Malcolm 2023-08-14) ?
        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # raise NotImplementedError

        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer_i, layer in enumerate(self.layers):
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only and layer_i < len(self.layers) - 1:
                inner_states.append(x)

        if self.upsample:
            x = x.transpose(0, 1)
            x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
            x = x.transpose(0, 1)

        inner_states.append(x)

        sentence_rep = x[0, :, :]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


class MusicBERTEncoder(FreezableRobertaEncoder):
    def __init__(self, args, dictionary, upsample: bool = True):
        super().__init__(args, dictionary)
        self.sentence_encoder = OctupleEncoder(
            upsample=upsample,
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )


@register_model("musicbert")
class MusicBERTModel(RobertaModel):
    encoder_cls = MusicBERTEncoder

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)  # modifies args in place
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        upsample = getattr(task, "upsample_encoder", True)

        encoder = cls.encoder_cls(args, task.source_dictionary, upsample=upsample)
        out = cls(args, encoder)  # type:ignore
        return out

    def register_sequence_tagging_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[  # type:ignore
                name
            ].out_proj.out_features  # type:ignore
            prev_inner_dim = self.classification_heads[  # type:ignore
                name
            ].dense.out_features  # type:ignore
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                LOGGER.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaSequenceTaggingHead(  # type:ignore
            input_dim=self.args.encoder_embed_dim,  # type:ignore
            inner_dim=inner_dim or self.args.encoder_embed_dim,  # type:ignore
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,  # type:ignore
            pooler_dropout=self.args.pooler_dropout,  # type:ignore
            q_noise=self.args.quant_noise_pq,  # type:ignore
            qn_block_size=self.args.quant_noise_pq_block_size,  # type:ignore
            do_spectral_norm=self.args.spectral_norm_classification_head,  # type:ignore
        )

    def register_multitask_sequence_tagging_head(
        self,
        name,
        num_classes: Sequence[int],
        inner_dim=None,
        encoder_embed_dim=None,
        **kwargs,
    ):
        """Register a classification head."""
        if encoder_embed_dim is None:
            encoder_embed_dim = self.args.encoder_embed_dim  # type:ignore
        if name in self.classification_heads:
            prev_num_classes = [
                x.out_proj.out_features
                for x in self.classification_heads[  # type:ignore
                    name
                ].multi_tag_sub_heads  # type:ignore
            ]
            prev_inner_dim_list = [
                x.dense.out_features
                for x in self.classification_heads[  # type:ignore
                    name
                ].multi_tag_sub_heads  # type:ignore
            ]
            assert len(set(prev_inner_dim_list)) == 1
            prev_inner_dim = prev_inner_dim_list[0]
            if num_classes != prev_num_classes or (
                inner_dim is not None and inner_dim != prev_inner_dim
            ):
                LOGGER.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        if name == "sequence_multitask_tagging_head":
            head_cls = RobertaSequenceMultiTaggingHead
        elif name == "sequence_multitask_conditional_tagging_head":
            head_cls = RobertaSequenceConditionalMultiTaggingHead
        else:
            raise ValueError

        self.classification_heads[  # type:ignore
            name
        ] = head_cls(
            input_dim=encoder_embed_dim,
            inner_dim=inner_dim or encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,  # type:ignore
            pooler_dropout=self.args.pooler_dropout,  # type:ignore
            q_noise=self.args.quant_noise_pq,  # type:ignore
            qn_block_size=self.args.quant_noise_pq_block_size,  # type:ignore
            do_spectral_norm=self.args.spectral_norm_classification_head,  # type:ignore
            liebel_loss=getattr(self.args, "liebel_loss", False),  # type:ignore
        )


@register_model_architecture("musicbert", "musicbert")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("musicbert", "musicbert_base")
def musicbert_base_architecture(args):
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_large")
def musicbert_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_medium")
def musicbert_medium_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_small")
def musicbert_small_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_mini")
def musicbert_mini_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_tiny")
def musicbert_tiny_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_test")
def musicbert_test_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 16)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    base_architecture(args)


class OctupleTokenDataset(PrependTokenDataset):
    def adaptor(self, e):
        # This function is used to convert the encoding to REMI or CP
        prev_bar = None
        prev_pos = None
        prev_prog = None
        new_e = []
        for i in e:
            if prev_bar != i[0]:
                prev_bar = i[0]
                prev_pos = None
                new_e.append((i[0], None, None, None, None, None, i[6], None))
            if prev_pos != i[1]:
                prev_pos = i[1]
                prev_prog = None
                new_e.append((None, i[1], None, None, None, None, None, i[7]))
            if prev_prog != i[2]:
                prev_prog = i[2]
                new_e.append((None, None, i[2], None, None, None, None, None))
            if True:
                new_e.append((None, None, None, i[3], i[4], i[5], None, None))
        return new_e

    def convert(self, item):
        # This function is used to convert the encoding to REMI or CP
        encoding = item[8:-8].tolist()
        encoding = list(tuple(encoding[i : i + 8]) for i in range(0, len(encoding), 8))
        encoding = self.adaptor(encoding)
        if CONVERT_ENCODING == "CP":
            encoding = list(3 if j is None else j for i in encoding for j in i)[
                : None if CROP_LENGTH is None else (CROP_LENGTH * 8)
            ]
        elif CONVERT_ENCODING == "REMI":
            encoding = list(j for i in encoding for j in i if j is not None)[
                :CROP_LENGTH
            ]
        else:
            assert False, "Unknown encoding format"
        bos = 0
        eos = 2
        encoding = ([bos] * 8) + encoding + ([eos] * 8)
        return torch.tensor(encoding)

    def __init__(self, dataset, token=None):
        super().__init__(dataset, token=None)
        if CONVERT_ENCODING != "OCTMIDI":
            self._sizes = np.array([len(self.convert(i)) for i in dataset])
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if CONVERT_ENCODING != "OCTMIDI":
            item = self.convert(item)
        return item

    def num_tokens(self, index):
        # Not so sure about this
        return self._sizes[index].item()

    def size(self, index):
        # Not so sure about this
        return self._sizes[index].item()


fairseq.tasks.sentence_prediction.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.MaskTokensDataset = OctupleMaskTokensDataset
