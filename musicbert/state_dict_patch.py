import logging
import re
from argparse import Namespace
from typing import Optional

from fairseq.checkpoint_utils import prune_state_dict
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from fairseq.models.roberta import RobertaModel
from omegaconf import DictConfig
from torch import nn

logger = logging.getLogger(__name__)


# For unknown reasons, load_state_dict is not moving the classification heads to CUDA.
# This patch to load_state_dict implements a hack to look after that.
def load_state_dict(
    self,
    state_dict,
    strict=True,
    model_cfg: Optional[DictConfig] = None,
    args: Optional[Namespace] = None,
):
    """Copies parameters and buffers from *state_dict* into this module and
    its descendants.

    Overrides the method in :class:`nn.Module`. Compared with that method
    this additionally "upgrades" *state_dicts* from old checkpoints.
    """

    if model_cfg is None and args is not None:
        logger.warn(
            "using 'args' is deprecated, please update your code to use dataclass config"
        )
        model_cfg = convert_namespace_to_omegaconf(args).model
    self.upgrade_state_dict(state_dict)
    new_state_dict = prune_state_dict(state_dict, model_cfg)

    # Replace call to super() with nn.Module:
    out = nn.Module.load_state_dict(self, new_state_dict, strict)

    # TODO: (Malcolm 2023-11-18) I'd like to be able to remove this
    # CUDA hack
    device = None
    for v in self.state_dict().values(): # NB state_dict != new_state_dict; 
        # new_state_dict is completely on cpu
        if v.device.type != "cpu":
            device = v.device
            break
    if device is not None:
        self.to(device)

    return out

RobertaModel.load_state_dict = load_state_dict


def upgrade_state_dict_named(self, state_dict, name):

    prefix = name + "." if name != "" else ""

    # rename decoder -> encoder before upgrading children modules
    for k in list(state_dict.keys()):
        if k.startswith(prefix + "decoder"):
            new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
            state_dict[new_k] = state_dict[k]
            del state_dict[k]

    # upgrade children modules
    super(RobertaModel, self).upgrade_state_dict_named(state_dict, name)  # type:ignore

    # Handle new classification heads present in the state dict.
    current_head_names = (
        []
        if not hasattr(self, "classification_heads")
        else self.classification_heads.keys()
    )
    keys_to_delete = []

    classes_per_target = {}

    for k in state_dict.keys():
        if not k.startswith(prefix + "classification_heads."):
            continue

        head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
        multitarget_head = head_name == "sequence_multitarget_tagging_head"

        if multitarget_head:
            # This actually a subhead
            m = re.search(
                r"sequence_multitarget_tagging_head\.multi_tag_sub_heads\.\d+",
                k[len(prefix + "classification_heads.") :],
            )
            assert m is not None
            head_name = m.group()

        # Original fairseq behavior
        num_classes = state_dict[
            prefix + "classification_heads." + head_name + ".out_proj.weight"
        ].size(0)
        inner_dim = state_dict[
            prefix + "classification_heads." + head_name + ".dense.weight"
        ].size(0)

        load_checkpoint_heads = getattr(self.args, "load_checkpoint_heads", False)

        if multitarget_head:
            assert load_checkpoint_heads, "--load-checkpoint-heads is required"
            if head_name not in classes_per_target:
                classes_per_target[head_name] = num_classes
            assert inner_dim == self.args.encoder_embed_dim
        else:
            if load_checkpoint_heads:
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
    for k in keys_to_delete:
        del state_dict[k]

    if classes_per_target:
        self.register_multitarget_sequence_tagging_head(
            "sequence_multitarget_tagging_head",
            num_classes=list(classes_per_target.values()),
        )

    # Copy any newly-added classification heads into the state dict
    # with their current weights.
    if hasattr(self, "classification_heads"):
        cur_state = self.classification_heads.state_dict()
        for k, v in cur_state.items():
            if prefix + "classification_heads." + k not in state_dict:
                logger.info("Overwriting " + prefix + "classification_heads." + k)
                # Hack to get these on to the right device
                device = self.state_dict()[prefix + "classification_heads." + k].device
                state_dict[prefix + "classification_heads." + k] = v.to(device)


RobertaModel.upgrade_state_dict_named = upgrade_state_dict_named
