"""
Implements a dual-encoder version of MusicBERT, which we use for predicting roman 
numerals conditional on keys.
"""

import logging
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.criterions import register_criterion
from fairseq.data import (
    NestedDictionaryDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.tasks import FairseqTask, register_task

from musicbert._musicbert import (
    MusicBERTEncoder,
    MusicBERTModel,
    musicbert_base_architecture,
)
from musicbert.token_classification_multi_task import (
    AssertSameLengthDataset,
    MultiTaskSequenceTaggingCriterion,
    MultiTaskSequenceTaggingTask,
)

LOGGER = logging.getLogger(__name__)

ACTIVATIONS = {"gelu": nn.GELU}


def mlp_layer(input_dim, output_dim, dropout, activation_fn, norm=True):
    modules: List[nn.Module] = [nn.Linear(input_dim, output_dim)]
    if dropout:
        modules.append(nn.Dropout(dropout))
    if norm:
        modules.append(nn.LayerNorm(output_dim))
    if activation_fn is not None:
        modules.append(ACTIVATIONS[activation_fn]())
    return nn.Sequential(*modules)


class MLP(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        activation_fn: str,
        norm: bool,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        assert n_layers > 0
        layers = []
        for _ in range(n_layers - 1):
            layers.append(
                mlp_layer(hidden_dim, hidden_dim, dropout, activation_fn, norm)
            )
        layers.append(mlp_layer(hidden_dim, output_dim, dropout, activation_fn, norm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        return self.layers(x)


class DualEncoder(MusicBERTEncoder):
    def __init__(self, args, dictionary, upsample: bool = True):
        super().__init__(args, dictionary, upsample=upsample)
        if args.z_encoder == "embedding":
            self.z_encoder = nn.Embedding(args.z_vocab_size, args.z_embed_dim)
        if args.z_encoder == "mlp":
            self.z_encoder = MLP(
                n_layers=args.z_mlp_layers,
                vocab_size=args.z_vocab_size,
                output_dim=args.z_embed_dim,
                hidden_dim=args.z_embed_dim,
                dropout=args.dropout,
                activation_fn=args.activation_fn,
                norm=args.z_mlp_norm == "yes",
            )
        else:
            raise ValueError

        if args.z_combine_procedure == "concat":
            self.combine_f = lambda x, z: torch.concat([x, z], dim=-1)
            self.output_dim = args.encoder_embed_dim + args.z_embed_dim
        elif args.z_combine_procedure == "project":
            self.combine_projection = nn.Linear(
                args.z_embed_dim + args.encoder_embed_dim, args.encoder_embed_dim
            )

            def combine_f(x, z):
                return self.combine_projection(torch.concat([x, z], dim=-1))

            self.combine_f = combine_f
            self.output_dim = args.encoder_embed_dim
        else:
            raise ValueError

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        freeze_encoder=False,
        z_tokens=None,
        **kwargs,
    ):
        assert features_only
        assert z_tokens is not None
        x, extra = super().forward(
            src_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            masked_tokens=masked_tokens,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )
        z = self.z_encoder(z_tokens)
        # xz = torch.concat([x, z], dim=-1)
        xz = self.combine_f(x, z)
        return xz, extra


@register_model("musicbert_dual_encoder")
class DualEncoderModel(MusicBERTModel):
    encoder_cls = DualEncoder

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        LOGGER.info(x["args"])
        return DualEncoderMusicBERTHubInterface(x["args"], x["task"], x["models"][0])


class DualEncoderMusicBERTHubInterface(RobertaHubInterface):
    # We need to subclass the parent class in order to provide conditioning in the
    # predict method

    def extract_features(
        self,
        tokens: torch.LongTensor,
        z_tokens: torch.LongTensor,
        return_all_hiddens: bool = False,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)  # type:ignore
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        if z_tokens.dim() == 1:
            z_tokens.unsqueeze(0)

        features, extra = self.model(
            tokens.to(device=self.device),  # type:ignore
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            z_tokens=z_tokens.to(device=self.device),  # type:ignore
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [
                inner_state.transpose(0, 1) for inner_state in inner_states
            ]  # type:ignore
        else:
            return features  # just the last layer's features

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))  # type:ignore
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)


@register_criterion("conditioned_multitask_sequence_tagging")
class ConditionedMultiTaskSequenceTaggingCriterion(MultiTaskSequenceTaggingCriterion):
    def __init__(self, task, classification_head_name):
        super().__init__(task, classification_head_name)

    @staticmethod
    def add_args(parser):
        MultiTaskSequenceTaggingCriterion.add_args(parser)

        parser.add_argument("--z-encoder", default="embedding", type=str)
        parser.add_argument("--z-embed-dim", default=128, type=int)
        parser.add_argument("--z-mlp-layers", default=2, type=int)
        parser.add_argument("--z-mlp-norm", default="yes", choices=["yes", "no"])
        parser.add_argument(
            "--z-combine-procedure", default="concat", choices=["concat", "project"]
        )

    def get_logits(self, model, sample):
        multi_logits, _ = model(
            **sample["net_input"],
            features_only=True,
            return_all_hiddens=False,
            classification_head_name=self.classification_head_name,
            z_tokens=sample["z_tokens"],
        )
        return multi_logits


@register_model_architecture("musicbert_dual_encoder", "musicbert_dual_encoder_base")
def musicbert_dual_encoder_architecture(args):
    musicbert_base_architecture(args)
    # TODO: (Malcolm 2024-03-02) set good default values
    args.z_embed_dim = getattr(args, "z_embed_dim", 128)
    args.z_vocab_size = getattr(args, "z_vocab_size", 128)
    args.z_mlp_layers = getattr(args, "z_mlp_layers", 2)
    args.z_mlp_norm = getattr(args, "z_mlp_norm", "yes")
    # z_combine_procedure: either "concat" or "project"
    args.z_combine_procedure = getattr(args, "z_combine_procedure", "concat")


@register_task("musicbert_conditioned_multitask_sequence_tagging")
class DualEncoderMultiTaskSequenceTagging(MultiTaskSequenceTaggingTask):

    @staticmethod
    def add_args(parser):
        MultiTaskSequenceTaggingTask.add_args(parser)

    def __init__(self, args, data_dictionary, label_dictionaries, cond_dictionary):
        super().__init__(args, data_dictionary, label_dictionaries)  # type:ignore
        self.cond_dictionary = cond_dictionary

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        temp_dataset = super().load_dataset(split, combine=combine, **kwargs)

        # we want to retrieve SortDataset -> NestedDictionaryDataset.defn
        assert isinstance(temp_dataset, SortDataset)
        sort_order = temp_dataset.sort_order
        temp_dataset = temp_dataset.dataset
        assert isinstance(temp_dataset, NestedDictionaryDataset)
        dataset_defn = temp_dataset.defn
        sizes = temp_dataset.sizes

        # dataset_defn should be a flattened dict[str, Dataset]
        assert isinstance(dataset_defn, dict)

        conditioning_dataset = self._load_dataset_helper_make_dataset(  # type:ignore
            "conditioning", self.cond_dictionary, split, combine
        )
        conditioning_dataset = RightPadDataset(
            conditioning_dataset, pad_idx=self.cond_dictionary.pad()
        )
        assert self.source_dictionary.pad() == self.cond_dictionary.pad()

        dataset_defn["z_tokens"] = conditioning_dataset
        temp_assert_lengths_match = dataset_defn.pop("_assert_lengths_match")
        assert_lengths_match = AssertSameLengthDataset(
            temp_assert_lengths_match.first,
            temp_assert_lengths_match.seconds + [conditioning_dataset],
            temp_assert_lengths_match.first_to_second_ratio,
        )

        dataset_defn["_assert_lengths_match"] = assert_lengths_match
        nested_dataset = NestedDictionaryDataset(dataset_defn, sizes)

        dataset = SortDataset(nested_dataset, sort_order=sort_order)

        self.datasets[split] = dataset
        return self.datasets[split]

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict, label_dicts = (
            MultiTaskSequenceTaggingTask._setup_task_helper(  # type:ignore
                args, **kwargs
            )
        )
        cond_dict = cls.load_dictionary(
            args, os.path.join(args.data, "conditioning", "dict.txt")  # type:ignore
        )
        LOGGER.info(f"[conditioning] dictionary: {len(cond_dict)} types")
        return cls(args, data_dict, label_dicts, cond_dict)

    def build_model(self, args):
        from fairseq import models

        args.z_vocab_size = len(self.cond_dictionary)
        model = models.build_model(args, self)
        self._build_model_freeze_helper(args, model)  # type:ignore
        num_classes = self._build_model_num_classes_helper()  # type:ignore

        model.register_multitask_sequence_tagging_head(
            getattr(
                args, "classification_head_name", "multitask_sequence_tagging_head"
            ),
            num_classes=num_classes,
            sequence_tagging=True,
            encoder_embed_dim=model.encoder.output_dim,
        )
        # We can't use the default fairseq checkpoint loading implementation because
        #   it uses strict=True which means that the encoder will cause the
        #   loading to fail

        # self._build_model_helper_load_checkpoint(model, args)

        return model
