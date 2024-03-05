import logging
import re

from fairseq.models.roberta import RobertaModel

logger = logging.getLogger(__name__)


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
            if "multi_tag_sub_heads" in k:
                # At a certain point I tried renaming _sub_heads while debugging,
                #   this line is here for compatibility.
                m = re.search(
                    r"sequence_multitarget_tagging_head\.multi_tag_sub_heads\.\d+",
                    k[len(prefix + "classification_heads.") :],
                )
            else:
                m = re.search(
                    r"sequence_multitarget_tagging_head\._sub_heads\.\d+",
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
            
            
            try:
                if self.args.z_combine_procedure == "concat":
                    assert inner_dim == self.args.encoder_embed_dim + self.args.z_embed_dim
                else:
                    assert inner_dim == self.args.encoder_embed_dim
            except AttributeError:
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

    # Copy any newly-added classification heads into the state dict
    # with their current weights.
    if hasattr(self, "classification_heads"):
        cur_state = self.classification_heads.state_dict()
        for k, v in cur_state.items():
            if prefix + "classification_heads." + k not in state_dict:
                logger.info("Overwriting " + prefix + "classification_heads." + k)
                state_dict[prefix + "classification_heads." + k] = v


RobertaModel.upgrade_state_dict_named = upgrade_state_dict_named
