import torch
import torch.nn.functional as F


def p_norm(x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0):
    return (x1 - x2).abs().pow(p).sum(axis=-1).pow(1 / p)  # type:ignore


def p_norm_loss(
    logits: torch.Tensor, p: float = 2.0, alpha: float = 0.01, reduction="mean"
):
    shape = logits.shape
    # We probably want to normalize the logits before calculating the norm since
    #   otherwise we are effectively regularizing them to be small (so the absolute
    #   differences between them are small), which we don't
    #   necessarily want. It seems that the appropriate way of doing this
    #   is just to take a softmax first.
    # logits have shape [..., seq, logits]
    probs = F.softmax(logits, dim=-1)
    norm = p_norm(probs[..., :-1, :], probs[..., 1:, :], p=p)
    # last dimension is gone; 2nd-last dimension reduced by 1
    assert norm.shape == (*shape[:-2], shape[-2] - 1)
    if reduction == "mean":
        norm = norm.mean()
    elif reduction == "sum":
        norm = norm.sum()
    else:
        raise NotImplementedError
    return alpha * norm


def cosine_sim_loss(logits: torch.Tensor, alpha: float = 0.01, reduction="mean"):
    # We probably don't want to use cosine similarity because vectors that are
    #   scaled versions of one another (e.g., [1, 2, 4], and [2, 4, 8]) have
    #   perfect cosine similarity but are quite different interpreted as logits
    raise NotImplementedError
