import torch
import torch.nn.functional as F


def p_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p: float = 2.0,
    min_eps: float = 1e-7,
    max_eps: float = 1e2,
):
    # this calculation is highly likely to overflow with fp16
    dtype = x1.dtype
    x1 = x1.float()
    x2 = x2.float()

    diffs = (x1 - x2).abs()
    # For numerical stability, we need to avoid infinities, which can happen
    #   if these differences are somewhat large (because they are then raised to *p*
    #   and then summed)
    diffs = torch.where(diffs > max_eps, max_eps, diffs)
    sum = diffs.pow(p).sum(axis=-1)  # type:ignore
    out = (sum + min_eps).pow(1 / p)
    return out.type(dtype)


def p_norm_loss(
    logits: torch.Tensor, p: float = 2.0, alpha: float = 0.01, reduction="mean"
):
    """
    An experimental loss function for penalizing changes in the distribution between
    successive time steps.
    """
    shape = logits.shape
    # We probably want to normalize the logits before calculating the norm since
    #   otherwise we are effectively regularizing them to be small (so the absolute
    #   differences between them are small), which we don't
    #   necessarily want. It seems that the appropriate way of doing this
    #   is just to take a softmax first.
    # logits have shape [..., seq, logits]
    probs = F.log_softmax(logits, dim=-1)
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
