"""Modeled on unmerged PR at https://github.com/facebookresearch/fairseq/pull/1710/files"""

import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.models.roberta import RobertaEncoder


@register_criterion("freezable_sentence_prediction")
class FreezableSentencePredictionCriterion(SentencePredictionCriterion):
    def __init__(
        self, task, classification_head_name, regression_target, freeze_encoder
    ):
        super().__init__(task, classification_head_name, regression_target)
        self.freeze_encoder = freeze_encoder

    @staticmethod
    def add_args(parser):
        SentencePredictionCriterion.add_args(parser)
        # fmt: off
        parser.add_argument('--freeze-encoder', action='store_true', default=False,
                            help='Freeze encoder weights and disable encoder dropout during training')
        # fmt: on

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
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
            freeze_encoder=self.freeze_encoder,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output


class FreezableRobertaEncoder(RobertaEncoder):
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        freeze_encoder=True,
        **unused
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            masked_tokens (BoolTensor, optional): tokens masked during
                masked LM training (default: None).
            freeze_encoder (bool, optional): freeze encoder weights (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        with torch.set_grad_enabled(not freeze_encoder):
            was_training = self.training
            if freeze_encoder:
                self.eval()  # disable dropout when encoder is frozen
            x, extra = self.extract_features(
                src_tokens, return_all_hiddens=return_all_hiddens
            )
            if was_training and freeze_encoder:
                self.train()
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra
