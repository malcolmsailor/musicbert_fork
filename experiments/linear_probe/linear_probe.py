"""
(Not sure the below commands still work after implementing wandb sweeps)

Local command:
python experiments/linear_probe/linear_probe.py \
    data_dir=~/output/test_data/labeled_chorales_bin \
    checkpoint=~/output/musicbert_checkpoints/32702693/checkpoint_best.pt \
    ref_dir=~/output/test_data/chord_tones_bin \
    debug=True
Grace command:
python experiments/linear_probe/linear_probe.py \
    data_dir=~/project/datasets/labeled_bach_chorales_bin \
    checkpoint=~/project/new_checkpoints/musicbert_fork/32702693/checkpoint_best.pt \
    ref_dir=~/project/datasets/chord_tones/fairseq/many_target_bin \
    debug=True

Wandb sweep:
    wandb sweep --project sweep-test linear_probe_config.yaml
Then:
    wandb agent msailor/sweep-test/v1uhkjcp

"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dacite import from_dict
from einops import rearrange, reduce, repeat
from fairseq.data import Dictionary, RightPadDataset, data_utils
from fairseq.models.roberta import RobertaModel
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb

# from musicbert._musicbert import OctupleEncoder

THIS_PATH = os.path.realpath(__file__)
DIRNAME, BASENAME = os.path.split(THIS_PATH)
WANDB_PROJECT = "dissonance-linear-probe"


def take_snapshot():
    with open(THIS_PATH) as inf:
        snapshot = inf.read()
    print("Took snapshot")
    return snapshot


if __name__ == "__main__":
    SNAPSHOT = take_snapshot()
else:
    SNAPSHOT = None

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

PARENT_DIR = os.path.join(DIRNAME, "..", "..")

sys.path.append(PARENT_DIR)

USER_DIR = os.path.join(PARENT_DIR, "musicbert")

SEED = 44
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data-dir",
#         required=True,
#         help="assumed to end in '_bin' and have an equivalent ending in '_raw' that contains 'metadata_test.txt'",
#     )
#     parser.add_argument(
#         "--ref-dir",
#         required=True,
#         help="points to the original data w/ target names and `label[x]/dict.txt` files, which is needed for checkpoint loading",
#         # help="a directory that contains `target_names.json` as well as "
#         # "`label[x]/dict.txt` files. If not provided, the value of "
#         # "--data-dir is used.",
#     )
#     parser.add_argument("--checkpoint", required=True)
#     parser.add_argument("--msdebug", action="store_true")

#     args = parser.parse_args()
#     return args


# @dataclass
# class ClassifierConfig:


@dataclass
class Config:
    data_dir: Optional[str] = None
    ref_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    debug: bool = False
    n_epochs: int = 10
    batch_size: int = 4
    layer_to_probe: int = 12
    loss_weights: Optional[Dict[str, float]] = field(
        default_factory=lambda: {"N": 2.0, "Z": 4.0}
    )
    early_stop_wait: int = 10
    early_stop_tolerance: float = 1e-2
    # classifier_config: ClassifierConfig = field(default_factory=ClassifierConfig)

    wandb_watch_freq: int = 50
    wandb_log_freq: int = 25

    # classifier config
    n_layers: int = 2
    input_dim: int = 768
    hidden_dim: int = 16

    def __post_init__(self):
        if self.data_dir is not None:
            self.data_dir = os.path.expanduser(self.data_dir)
        if self.checkpoint is not None:
            self.checkpoint = os.path.expanduser(self.checkpoint)
        if self.ref_dir is not None:
            self.ref_dir = os.path.expanduser(self.ref_dir)


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


def classifier_layer(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())


class Classifier(nn.Module):
    def __init__(self, output_dim, config: Config):
        super().__init__()
        assert config.n_layers >= 2
        self.layers = nn.Sequential(
            classifier_layer(config.input_dim, config.hidden_dim),
            *(
                classifier_layer(config.hidden_dim, config.hidden_dim)
                for _ in range(config.n_layers - 2)
            ),
            nn.Linear(config.hidden_dim, output_dim),
        )

    def forward(self, x):
        leading_dims = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.layers(x)
        x = x.reshape(*leading_dims, -1)
        return x


def read_config_oc(config_cls, yaml_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=None)
    # remaining passed through to omegaconf

    args, remaining = parser.parse_known_args()

    configs = []
    assert args.config_path is not None or remaining is not None
    if yaml_path is not None:
        configs.append(OmegaConf.load(yaml_path))
    if args.config_path is not None:
        configs.append(OmegaConf.load(args.config_path))
    if remaining is not None:
        configs.append(OmegaConf.from_cli(remaining))
    merged_conf = OmegaConf.merge(*configs)
    resolved = OmegaConf.to_container(merged_conf, resolve=True)
    assert isinstance(resolved, dict)
    out = from_dict(data_class=config_cls, data=resolved)  # type:ignore
    return out


def get_data(data_dir):
    label_dict = Dictionary.load(os.path.join(data_dir, "label", "dict.txt"))
    labels = {}
    for split in ("train", "valid", "test"):
        split_path = os.path.join(data_dir, "label", split)
        if os.path.exists(f"{split_path}.bin"):
            dataset = data_utils.load_indexed_dataset(split_path, label_dict)
            dataset = RightPadDataset(dataset, pad_idx=label_dict.pad())
            labels[split] = dataset
    return label_dict, labels


def get_y_and_y_hat(
    ds, labels_ds, indices, encoder, classifier, train_config: Config, n_specials
):
    samples = [ds[j] for j in indices]
    labels = [labels_ds[j] for j in indices]
    batch_dict = ds.collater(samples)
    batch_input = batch_dict["net_input"]["src_tokens"].to(DEVICE)
    y = labels_ds.collater(labels).to(DEVICE)
    inner_states, sentence_rep = encoder(batch_input)
    probe_states = inner_states[train_config.layer_to_probe]
    probe_states = rearrange(probe_states, "seq batch d_model -> batch seq d_model")
    y_hat = classifier(probe_states)

    # ignore specials
    y -= n_specials
    y = torch.where(y < 0, -1, y)
    return y, y_hat


def avg_list(l):
    return sum(l) / len(l)


def train(
    encoder,
    classifier: Classifier,
    optim: torch.optim.Optimizer,
    datasets,
    label_datasets,
    train_config: Config,
    n_specials: int,
    loss_weights: Optional[torch.Tensor] = None,
    shuffle=True,
):
    metrics = defaultdict(lambda: defaultdict(list))
    steps = defaultdict(list)
    for split in ("test", "valid", "train"):
        if split in datasets:
            assert len(datasets[split]) == len(label_datasets[split])

    early_stopper = EarlyStopper(
        train_config.early_stop_wait, train_config.early_stop_tolerance
    )

    # Freeze encoder
    encoder.eval()
    result = "timeout"
    training_step = 0
    wandb.watch(classifier, log="all", log_freq=train_config.wandb_watch_freq)

    train_ds = datasets["train"]
    train_labels = label_datasets["train"]

    valid_ds = datasets["valid"]
    valid_labels = label_datasets["valid"]

    if loss_weights is not None:
        loss_weights = loss_weights.to(DEVICE)

    try:
        for epoch_i in range(train_config.n_epochs):
            indices = list(range(len(train_ds)))
            if shuffle:
                random.shuffle(indices)
            pbar = tqdm(range(0, len(train_ds), train_config.batch_size))
            pbar.set_description(f"Epoch {epoch_i + 1}/{train_config.n_epochs} train")
            classifier.train()
            for i in pbar:
                optim.zero_grad()

                y, y_hat = get_y_and_y_hat(
                    train_ds,
                    train_labels,
                    indices[i : i + train_config.batch_size],
                    encoder,
                    classifier,
                    train_config,
                    n_specials,
                )
                loss = F.cross_entropy(
                    rearrange(y_hat, "... logits -> (...) logits"),
                    rearrange(y, "... -> (...)"),
                    ignore_index=-1,
                    weight=loss_weights,
                )
                loss.backward()
                metrics["loss"]["train"].append(loss.item())
                optim.step()
                pbar.set_postfix({"loss": avg_list(metrics["loss"]["train"])})
                training_step += 1
                steps["train"].append(training_step)
                if training_step % train_config.wandb_log_freq == 0:
                    wandb.log({"train_loss": loss.item()}, step=training_step)

            classifier.eval()
            pbar = tqdm(range(0, len(valid_ds), train_config.batch_size))
            pbar.set_description(f"Epoch {epoch_i + 1}/{train_config.n_epochs} valid")
            valid_loss = []
            with torch.no_grad():
                for i in pbar:
                    indices = range(i, min(len(valid_ds), i + train_config.batch_size))
                    y, y_hat = get_y_and_y_hat(
                        valid_ds,
                        valid_labels,
                        indices,
                        encoder,
                        classifier,
                        train_config,
                        n_specials,
                    )
                    loss = F.cross_entropy(
                        rearrange(y_hat, "... logits -> (...) logits"),
                        rearrange(y, "... -> (...)"),
                        ignore_index=-1,
                        weight=loss_weights,
                    )
                    valid_loss.append(loss.item())
                    pbar.set_postfix({"valid_loss": avg_list(valid_loss)})
            epoch_valid_loss = avg_list(valid_loss)
            wandb.log({"valid_loss": epoch_valid_loss}, step=training_step)
            metrics["loss"]["valid"].append(epoch_valid_loss)
            steps["valid"].append(training_step)
            early_stopper.update(epoch_valid_loss)
            if early_stopper.stop():
                print(f"Reached early-stopping threshold")
                break

        result = "end"
    except KeyboardInterrupt:
        save = input("Save logs y/n? ").lower()[0] == "y"
        result = "fail"
    else:
        save = True
    return result, save, metrics, steps


def init(config: Config):
    # args = parse_args()

    assert config.checkpoint is not None
    assert config.ref_dir is not None
    assert config.data_dir is not None

    if config.debug:
        import pdb
        import sys
        import traceback

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    with open(os.path.join(config.ref_dir, "target_names.json"), "r") as inf:
        target_names = json.load(inf)

    musicbert = RobertaModel.from_pretrained(
        model_name_or_path=PARENT_DIR,
        checkpoint_file=config.checkpoint,
        data_name_or_path=config.data_dir,
        user_dir=USER_DIR,
        task="musicbert_multitarget_sequence_tagging",
        ref_dir=config.ref_dir,
        target_names=target_names,
    )

    label_dict, labels = get_data(config.data_dir)
    n_specials = 0
    n_tokens = 0
    for symbol in label_dict.symbols:
        if symbol.startswith("<") and symbol.endswith(">"):
            n_specials += 1
        elif not symbol.startswith("madeupword"):
            n_tokens += 1

    if config.loss_weights is not None:
        loss_weights = []
        for i in range(n_tokens):
            symbol = label_dict.symbols[n_specials + i]
            loss_weights.append(config.loss_weights.get(symbol, 1.0))
        loss_weights = torch.tensor(loss_weights)
    else:
        loss_weights = None

    if torch.cuda.is_available():
        musicbert.cuda()

    musicbert.eval()
    encoder = musicbert.model.encoder.sentence_encoder  # type:ignore

    musicbert.task.load_dataset("train")
    musicbert.task.load_dataset("valid")
    # musicbert.task.load_dataset("test")
    classifier = Classifier(n_tokens, config)
    classifier.to(DEVICE)
    optim = torch.optim.Adam(classifier.parameters())

    datasets = musicbert.task.datasets
    return (
        encoder,
        classifier,
        optim,
        datasets,
        labels,
        config,
        n_specials,
        loss_weights,
    )
    # train(
    #     encoder,
    #     classifier,
    #     optim,
    #     datasets=musicbert.task.datasets,
    #     label_datasets=labels,
    #     train_config=train_config,
    #     n_specials=n_specials,
    #     loss_weights=loss_weights,
    # )

    # dummy_tokens = torch.arange(64).reshape((1, -1))
    # inner_states, sentence_rep = encoder(dummy_tokens)  # type:ignore

    # x = classifier(inner_states[-4])
    # breakpoint()


def save_snapshot(run_id, snapshot):
    # Start the snapshot with a non-integer character so we can import it as a
    # Python module
    snapshot_path = os.path.join(DIRNAME, "snapshots", f"s{run_id}_{BASENAME}")
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    # shutil.copy2(THIS_PATH, snapshot_path)
    with open(snapshot_path, "w") as outf:
        outf.write(snapshot)
    print(f"Saved {snapshot_path}")


def plot_metrics(
    run_id,
    result,
    metrics,
    steps,
):
    fig, ax = plt.subplots(nrows=len(metrics), squeeze=False)
    ax = ax.flatten()
    for plot_i, metric_name in enumerate(metrics):
        for kind, values in metrics[metric_name].items():

            ax[plot_i].plot(steps.get(kind, np.arange(len(values))), values, label=kind)
        ax[plot_i].set_title(metric_name)
        ax[plot_i].set_xlabel("Training step")
        ax[plot_i].legend()

    fig.tight_layout()
    root, _ = os.path.splitext(BASENAME)
    fig_path = os.path.join(DIRNAME, "logs", f"{run_id}_{root}_{result}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")


def save_checkpoint(model, run_id):
    root, _ = os.path.splitext(BASENAME)
    checkpoint_path = os.path.join(DIRNAME, "checkpoints", f"{run_id}_{root}.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(obj=model.state_dict(), f=checkpoint_path)


# def objective(wandb_config):
#     config: Config = from_dict(data_class=Config, data=wandb_config)

#     (
#         encoder,
#         classifier,
#         optim,
#         datasets,
#         labels,
#         train_config,
#         n_specials,
#         loss_weights,
#     ) = init(config)

#     train(
#         encoder,
#         classifier,
#         optim,
#         datasets,
#         labels,
#         train_config,
#         n_specials,
#         loss_weights,
#     )


def run_as_script():
    run_id = int(time.time())

    save = True

    temp_config = read_config_oc(Config)
    wandb.login()
    with wandb.init(project=WANDB_PROJECT, config=asdict(temp_config), dir="/home/ms3682/tmp"):  # type:ignore
        config_dict = wandb.config
        config = from_dict(data_class=Config, data=config_dict)

        (
            encoder,
            classifier,
            optim,
            datasets,
            labels,
            train_config,
            n_specials,
            loss_weights,
        ) = init(config)

        (
            result,
            save,
            metrics,
            steps,
        ) = train(
            encoder,
            classifier,
            optim,
            datasets,
            labels,
            train_config,
            n_specials,
            loss_weights,
        )

        if save:
            save_snapshot(run_id, SNAPSHOT)
            plot_metrics(
                run_id,
                result,
                metrics,
                steps,
            )
            save_checkpoint(classifier, run_id)


if __name__ == "__main__":
    run_as_script()
