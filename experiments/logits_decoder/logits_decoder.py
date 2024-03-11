import math
import random
import sys
from dataclasses import asdict, dataclass
from typing import Optional
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.utils.data import Dataset, DataLoader
import h5py
import re

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import wandb
import sklearn.metrics

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

import traceback, pdb, sys


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


@dataclass
class Config:
    train_logits_h5: str
    train_label_h5: str
    # valid_logits_h5: str
    # test_logits_h5: str
    input_dim: int = 24
    d_model: int = 24
    n_head: int = 2
    d_ff: int = 32
    dropout: float = 0.1
    n_layers: int = 2
    output_dim: int = 24

    segment_len: int = 100
    segment_overlap: int = 25

    sinusoidal_pe: bool = True
    pe_dropout: Optional[float] = None

    n_epochs: int = 30
    batch_size: int = 32

    lr: float = 1e-3
    lr_scheduler: str = "constant"
    lr_warmup: bool = False
    lr_warmup_steps: int = 100

    early_stop_wait: int = 10
    early_stop_tolerance: float = 1e-2

    wandb_project: str = "scratch"  # TODO: (Malcolm 2024-03-11)
    wandb_watch_freq: int = 50
    wandb_log_freq: int = 25

    seed: int = 42

    debug: bool = False

    def __post_init__(self):
        if self.pe_dropout is None:
            self.pe_dropout = self.dropout


class LogitsData(Dataset):
    def __init__(self, logits_h5: str, labels_h5, config: Config):
        super().__init__()
        self._raw_logits = []
        self._logits_indices = []
        self._segment_len = config.segment_len
        self._raw_labels = []
        with h5py.File(logits_h5) as logits_h5file:
            with h5py.File(labels_h5) as labels_h5file:
                for i, logits_key in enumerate(
                    sorted(
                        logits_h5file.keys(),
                        key=lambda x: int(re.search(r"\d+", x).group()),  # type:ignore
                    )
                ):
                    assert logits_key == f"logits_{i}"
                    labels_key = f"labels_{i}"
                    logits: np.ndarray = (
                        logits_h5file[logits_key]
                    )[  # type:# type:ignore
                        :
                    ]
                    labels: np.ndarray = (
                        labels_h5file[labels_key]
                    )[  # type:# type:ignore
                        :
                    ]
                    assert labels.shape[0] == logits.shape[0]

                    self._raw_logits.append(torch.tensor(logits).float())
                    self._raw_labels.append(torch.tensor(labels))
                    for j in range(
                        0, logits.shape[0], config.segment_len - config.segment_overlap
                    ):
                        self._logits_indices.append((i, j))
                        if j + config.segment_len > logits.shape[0]:
                            break

    def __getitem__(self, index):
        # TODO: x and y (Malcolm 2024-03-11)
        i, j = self._logits_indices[index]
        logits = self._raw_logits[i][j : j + self._segment_len]
        labels = self._raw_labels[i][j : j + self._segment_len]
        return logits, labels

    def __len__(self):
        return len(self._logits_indices)


def collate_fn(batch):
    # TODO: (Malcolm 2024-03-11) I'm not sure what the correct approach to padding is
    #   here
    logits, labels = zip(*batch)
    logits = pad_sequence(logits, batch_first=True, padding_value=0.0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return logits.to(DEVICE), labels.to(DEVICE)


class SinusoidalPositionalEncoding(nn.Module):
    # after https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(
        self, d_model: int, dropout: Optional[float] = 0.1, max_len: int = 5000
    ):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]  # type:ignore
        return self.dropout(x)


class Transformer1(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )
        decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.n_layers)
        self.decoder = decoder

        self.output_head = nn.Linear(
            in_features=config.d_model, out_features=config.output_dim
        )
        if config.sinusoidal_pe:
            self.pe = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.segment_len,
                dropout=config.pe_dropout,
            )
        else:
            embed = nn.Embedding(config.segment_len, config.d_model)
            if config.pe_dropout:
                self.pe = nn.Sequential(embed, nn.Dropout(config.pe_dropout))
            else:
                self.pe = embed

    def forward(self, x: torch.Tensor):
        x = rearrange(x, "batch seq ... -> seq batch ...")
        x = self.pe(x)
        src_mask = nn.Transformer.generate_square_subsequent_mask(len(x)).to(x.device)
        x = self.decoder(x, mask=src_mask, is_causal=True)
        x = self.output_head(x)
        x = rearrange(x, "seq batch ... -> batch seq ...")
        return x


class EarlyStopper:
    def __init__(self, wait, tolerance):
        self.wait = wait
        assert tolerance >= 0
        self.tolerance = tolerance
        self._count = 0
        self._last_min_loss = float("inf")

    def update(self, loss):
        if self._last_min_loss > loss:
            self._last_min_loss = loss
        if self._last_min_loss + self.tolerance > loss:
            self._count = 0
        else:
            self._count += 1

    def stop(self):
        if self._count >= self.wait:
            return True
        return False


def get_lr_scheduler(optim, total_steps, config: Config):
    if config.lr_warmup:
        total_steps -= config.lr_warmup_steps
    if config.lr_scheduler == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor=1.0)
    elif config.lr_scheduler == "cosine_annealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps
        )
    elif config.lr_scheduler == "linear_decay":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    else:
        raise ValueError

    if config.lr_warmup:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1e-3, end_factor=1.0
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim, [warmup_scheduler, lr_scheduler], milestones=[config.lr_warmup_steps]
        )
    return lr_scheduler


def get_total_training_steps(config, train_data):
    return math.ceil(len(train_data) / config.batch_size) * config.n_epochs


def avg_list(l):
    return sum(l) / len(l)


def log_scalar(key, val, step=None):
    wandb.log({key: val}, step=step)


def log_metrics(y_true, y_pred, step, labels=("x", "P", "S"), kind=""):
    no_specials_mask = y_true >= 0  # type:ignore
    y_true = y_true[no_specials_mask]
    y_pred = y_pred[no_specials_mask]
    for average in ["micro", "weighted", "macro"]:
        (
            precision,
            recall,
            f1,
            support,
        ) = sklearn.metrics.precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0.0  # type:ignore
        )
        log_scalar(f"{f'{kind}_' if kind else ''}precision_{average}", precision, step)
        log_scalar(f"{f'{kind}_' if kind else ''}recall_{average}", recall, step)
        log_scalar(f"{f'{kind}_' if kind else ''}f1_{average}", f1, step)

    confused = sklearn.metrics.confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_per_class = np.nan_to_num(confused.diagonal() / confused.sum(axis=0))
        recall_per_class = np.nan_to_num(confused.diagonal() / confused.sum(axis=1))
        f1_per_class = np.nan_to_num(
            (2 * precision_per_class * recall_per_class)
            / (precision_per_class + recall_per_class)
        )
    for (
        label_i,
        label,
    ) in enumerate(labels):
        # specials may or may not be included in the metric arrays, but we
        #   don't want to log them. So instead we do as follows:
        class_i = len(precision_per_class) - len(labels) + label_i
        log_scalar(
            f"{f'{kind}_' if kind else ''}precision_{label}",
            precision_per_class[class_i],
            step,
        )
        log_scalar(
            f"{f'{kind}_' if kind else ''}recall_{label}",
            recall_per_class[class_i],
            step,
        )
        log_scalar(
            f"{f'{kind}_' if kind else ''}f1_{label}", f1_per_class[class_i], step
        )


def train(train_data, valid_data, model: Transformer1, config: Config):
    train_dl = DataLoader(train_data, config.batch_size, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_data, config.batch_size, collate_fn=collate_fn)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_lr_scheduler(
        optim, get_total_training_steps(config, train_data), config
    )
    training_step = 0
    early_stopper = EarlyStopper(config.early_stop_wait, config.early_stop_tolerance)

    for epoch_i in range(config.n_epochs):
        print(f"Training {epoch_i + 1}/{config.n_epochs}")
        model.train()
        for batch_i, batch in (
            pbar := tqdm(enumerate(train_dl), total=len(train_dl), ncols=80)
        ):
            optim.zero_grad()
            x, y = batch
            logits = model(x)
            loss = loss_fn(
                rearrange(logits, "... n_classes -> (...) n_classes"),
                rearrange(y, "... -> (...)"),
            )
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.set_postfix({"loss": loss.item()})
            training_step += 1
            if training_step % config.wandb_log_freq == 0:
                wandb.log({"train_loss": loss.item()}, step=training_step)

        model.eval()
        valid_loss = []
        y_true_accumulator = []
        y_pred_accumulator = []

        with torch.no_grad():
            for batch_i, batch in (
                pbar := tqdm(enumerate(valid_dl), total=len(valid_dl), ncols=80)
            ):
                x, y = batch
                logits = model(x)
                loss = loss_fn(
                    rearrange(logits, "... n_classes -> (...) n_classes"),
                    rearrange(y, "... -> (...)"),
                )
                valid_loss.append(loss.item())
                pbar.set_postfix({"valid_loss": avg_list(valid_loss)})
                y_true_accumulator.append(y.detach().cpu().numpy())
                y_pred_accumulator.append(logits.argmax(dim=-1).detach().cpu().numpy())
        epoch_valid_loss = avg_list(valid_loss)
        early_stopper.update(epoch_valid_loss)
        wandb.log({"valid_loss": epoch_valid_loss}, step=training_step)
        y_true = np.concatenate(
            [rearrange(y, "... -> (...)") for y in y_true_accumulator]
        )
        y_pred = np.concatenate(
            [rearrange(y, "... -> (...)") for y in y_pred_accumulator]
        )
        log_metrics(y_true, y_pred, training_step, kind="valid")
        if early_stopper.stop():
            print(f"Reached early-stopping threshold")
            break


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    if config.debug:
        sys.excepthook = custom_excepthook

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_data = LogitsData(config.train_logits_h5, config.train_label_h5, config)
    # TODO: (Malcolm 2024-03-11)
    valid_data = LogitsData(config.train_logits_h5, config.train_label_h5, config)
    model = Transformer1(config)
    model.to(DEVICE)

    # x, y = next(iter(train_dl))
    wandb.login()
    with wandb.init(
        project=config.wandb_project, config=asdict(config)  # type:ignore
    ):
        train(train_data, valid_data, model, config)

    # data = torch.rand((4, config.ctx_length, config.d_model))
    # y = model(data)
    # breakpoint()


if __name__ == "__main__":
    main()
