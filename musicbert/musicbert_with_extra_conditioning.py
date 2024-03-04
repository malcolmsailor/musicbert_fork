import logging
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from fairseq.tasks import FairseqTask, register_task

from musicbert._musicbert import (
    MusicBERTEncoder,
    MusicBERTModel,
    musicbert_base_architecture,
)
from musicbert.token_classification_multi_target import (
    AssertSameLengthDataset,
    MultiTargetSequenceTaggingCriterion,
    MultiTargetSequenceTaggingTask,
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
                norm=args.z_mlp_norm,
            )

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
        xz = torch.concat([x, z], dim=-1)
        return xz, extra


@register_model("musicbert_dual_encoder")
class DualEncoderModel(MusicBERTModel):
    # TODO: (Malcolm 2024-03-01) use register_multitarget_sequence_tagging_head with
    # encoder_embed_dim set appropriately
    encoder_cls = DualEncoder

    # def forward(
    #     self,
    #     src_tokens,
    #     features_only=False,
    #     return_all_hiddens=False,
    #     classification_head_name=None,
    #     z_tokens=None,
    #     **kwargs,
    # ):
    #     breakpoint()


@register_criterion("conditioned_multitarget_sequence_tagging")
class ConditionedMultiTargetSequenceTaggingCriterion(
    MultiTargetSequenceTaggingCriterion
):
    def __init__(self, task, classification_head_name):
        super().__init__(task, classification_head_name)

    @staticmethod
    def add_args(parser):
        MultiTargetSequenceTaggingCriterion.add_args(parser)
        # fmt: off
        parser.add_argument('--z-encoder',default="embedding", type=str)
        # fmt: on

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
    args.z_mlp_norm = getattr(args, "z_mlp_norm", True)


@register_task("musicbert_conditioned_multitarget_sequence_tagging")
class DualEncoderMultiTargetSequenceTagging(MultiTargetSequenceTaggingTask):

    @staticmethod
    def add_args(parser):
        MultiTargetSequenceTaggingTask.add_args(parser)

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
            MultiTargetSequenceTaggingTask._setup_task_helper(  # type:ignore
                args, **kwargs
            )
        )
        cond_dict = cls.load_dictionary(
            args, os.path.join(args.data, "conditioning", "dict.txt")  # type:ignore
        )
        LOGGER.info(f"[conditioning] dictionary: {len(cond_dict)} types")
        return cls(args, data_dict, label_dicts, cond_dict)

    # TODO: (Malcolm 2024-03-04) remove in favor of monkeypatch
    # def _build_model_helper_load_checkpoint(self, model, args):
    #     if not args.restore_file:
    #         return
    #     # Load the original pretrained model
    #     from fairseq.checkpoint_utils import load_model_ensemble

    #     models, _ = load_model_ensemble(
    #         [args.restore_file],
    #         task=MultiTargetSequenceTaggingTask(
    #             args, self.dictionary, self._label_dictionaries  # type:ignore
    #         ),
    #     )
    #     pretrained_model = models[0]

    #     # Transfer weights from the pretrained model
    #     model_dict = model.state_dict()
    #     for name, param in pretrained_model.named_parameters():
    #         if name in model_dict:
    #             model_dict[name].copy_(param.data)
    #         else:
    #             print(f"Skipping {name} as it's not in the custom model")

    #     # Now remove args.restor_file so we don't try to load the checkpoint later
    #     args.restore_file = None

    def build_model(self, args):
        from fairseq import models

        args.z_vocab_size = len(self.cond_dictionary)
        model = models.build_model(args, self)
        self._build_model_freeze_helper(args, model)  # type:ignore
        num_classes = self._build_model_num_classes_helper()  # type:ignore
        # TODO: (Malcolm 2024-03-02)

        model.register_multitarget_sequence_tagging_head(
            getattr(
                args, "classification_head_name", "multitarget_sequence_tagging_head"
            ),
            num_classes=num_classes,
            sequence_tagging=True,
            encoder_embed_dim=args.encoder_embed_dim + args.z_embed_dim,
        )
        # We can't use the default fairseq checkpoint loading implementation because
        #   it uses strict=True which means that the encoder will cause the
        #   loading to fail

        # self._build_model_helper_load_checkpoint(model, args)

        return model
