import torch


def p_norm(x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0):
    return (x1 - x2).abs().pow(p).sum(axis=-1).pow(1 / p)  # type:ignore


def p_norm_loss(
    logits: torch.Tensor, p: float = 2.0, alpha: float = 0.01, reduction="mean"
):
    shape = logits.shape
    # logits have shape [..., seq, logits]
    norm = p_norm(logits[..., :-1, :], logits[..., 1:, :], p=p)
    # last dimension is gone; 2nd-last dimension reduced by 1
    assert norm.shape == (*shape[:-2], shape[-2] - 1)
    if reduction == "mean":
        norm = norm.mean()
    elif reduction == "sum":
        norm = norm.sum()
    else:
        raise NotImplementedError
    return alpha * norm
