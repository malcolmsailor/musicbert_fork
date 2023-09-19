import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from itertools import count


def shell(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True).stdout.decode()


# For the sake of reproducibility, we want to commit all changes in the directory before
# running. Then we can get the git hash and command from wandb to reproduce (hopefully!)

if not os.getenv("DEBUG_MUSICBERT", None):
    uncommited_changes = shell("git status --porcelain")
    if uncommited_changes:
        print("There are uncommitted changes; commit them then rerun")
        sys.exit(1)

if shutil.which("nvidia-smi"):
    nvidia_out = shell("nvidia-smi --query-gpu=name --format=csv,noheader")
    N_GPU_LOCAL = len(nvidia_out.strip().split("\n"))
    CPU_FLAG = ""
else:
    N_GPU_LOCAL = 0
    CPU_FLAG = "--cpu"

TOTAL_UPDATES = 125000
WARMUP_UPDATES = 25000

DEFAULT_CHECKPOINT = os.getenv(
    "MUSICBERT_DEFAULT_CHECKPOINT", 
    os.path.join(
        os.environ["SAVED_CHECKPOINTS_DIR"], 
        "/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt"
    )
)

PEAK_LR=0.0005 # Borrowed from musicbert

# NB in musicbert scripts, UPDATE_BATCH_SIZE is only used in the UPDATE_FREQ calculation below;
#   the actual batch size to fairseq-train is set by BATCH_SIZE arg
UPDATE_BATCH_SIZE = 64
BATCH_SIZE = 4


SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
USER_DIR = os.path.join(SCRIPT_DIR, "../musicbert")


parser = argparse.ArgumentParser()
parser.add_argument("--data-bin-dir", "-d", required=True)
parser.add_argument("--architecture", "-a", required=True)
parser.add_argument("--wandb-project", "-W", required=True)
parser.add_argument("--total-updates", "-u", type=int, default=TOTAL_UPDATES)
parser.add_argument("--warmup-updates", "-w", type=int, default=WARMUP_UPDATES)
parser.add_argument("--update-batch-size", type=int, default=UPDATE_BATCH_SIZE)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--lr", type=float, default=PEAK_LR)
parser.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
parser.add_argument("--multitarget", action="store_true")
args, args_to_pass_on = parser.parse_known_args()


TOKENS_PER_SAMPLE = 8192


UPDATE_FREQ_DENOM = max(N_GPU_LOCAL, 1)
UPDATE_FREQ = min(args.update_batch_size // (args.batch_size * UPDATE_FREQ_DENOM), 1)

NEW_CHECKPOINTS_DIR = os.environ["NEW_CHECKPOINTS_DIR"]

SLURM_ID = os.getenv("SLURM_JOB_ID", None)

if SLURM_ID is not None:
    # We're running a Slurm job
    SAVE_DIR = os.path.join(NEW_CHECKPOINTS_DIR, "musicbert_fork", SLURM_ID)
    # We can't use os.cpu_count() because it will show all the cpus on the node
    #   rather than just those allocated to our job
    CPUS_ON_NODE = os.getenv("SLURM_CPUS_ON_NODE", 1)
else:
    # We're not running a Slurm job
    SAVE_DIR = os.path.join(
        NEW_CHECKPOINTS_DIR, "musicbert_fork", str(round(time.time()))
    )
# TODO: (Malcolm 2023-09-19) restore
CPUS_ON_NODE = os.cpu_count()

os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)

DATA_BIN_DIR = args.data_bin_dir
NN_ARCH = f"musicbert_{args.architecture}"
WANDB_PROJECT = args.wandb_project

RESTORE_FLAG = "" if not args.checkpoint else f"--restore-file {args.checkpoint}"

if args.multitarget:
    num_classes = []
    for i in count():
        label_dict_file = os.path.join(DATA_BIN_DIR, f"label{i}", "dict.txt")
        if not os.path.exists(label_dict_file):
            break
        with open(label_dict_file, "r") as label_file:
            num_classes.append(
                str(
                    len(
                        [
                            line
                            for line in label_file
                            if not re.match(r"madeupword[0-9]{4}", line)
                        ]
                    )
                )
            )
    NUM_CLASSES = " ".join(num_classes)
else:
    label_dict_file = os.path.join(DATA_BIN_DIR, "label", "dict.txt")
    with open(label_dict_file, "r") as label_file:
        NUM_CLASSES = len(
            [line for line in label_file if not re.match(r"madeupword[0-9]{4}", line)]
        )

TASK = (
    "musicbert_multitarget_sequence_tagging"
    if args.multitarget
    else "musicbert_sequence_tagging"
)

HEAD_NAME = (
    "sequence_multitarget_tagging_head" if args.multitarget else "sequence_tagging_head"
)
CRITERION = "multitarget_sequence_tagging" if args.multitarget else "sequence_tagging"

ARGS = (
    " ".join(
        [
            DATA_BIN_DIR,
            CPU_FLAG,
            f"--user-dir {USER_DIR}",
            RESTORE_FLAG,
            f"--save-dir {SAVE_DIR}",
            f"--wandb-project {WANDB_PROJECT}",
            f"--task {TASK}",
            f"--arch {NN_ARCH}",
            f"--batch-size {args.batch_size}",
            f"--update-freq {UPDATE_FREQ}",
            f"--criterion {CRITERION}",
            f"--classification-head-name {HEAD_NAME}",
            "--compound-token-ratio 8",
            f"--num-classes {NUM_CLASSES}",
            # These `reset` params seem to be required for fine-tuning
            "--reset-optimizer",
            "--reset-dataloader",
            "--reset-meters",
            # Most of following hyperparameters directly from musicbert scripts
            # (they seem to be themselves borrowed from fairseq tutorial)
            "--optimizer adam",
            "--adam-betas (0.9,0.98)",
            "--adam-eps 1e-6",
            "--clip-norm 0.0",
            "--lr-scheduler polynomial_decay",
            f"--lr {args.lr}",
            "--log-format simple",
            f"--warmup-updates {args.warmup_updates}",
            f"--total-num-update {args.total_updates}",
            f"--max-update {args.total_updates}",
            "--no-epoch-checkpoints",
            "--find-unused-parameters",
            # TODO: (Malcolm 2023-08-29) update best checkpoint metric (f1?)
            "--best-checkpoint-metric accuracy",
            "--maximize-best-checkpoint-metric",
            # I believe we need to keep max positions the same as musicbert
            "--max-positions 8192",
            "--required-batch-size-multiple 1",
            # --shorten-method, --init-token, and --separator-token are unrecognized
            #   arguments for token classification. TODO I should inspect this further.
            # --shorten-method 'truncate'
            # --init-token 0 --separator-token 2
            #
            # TODO: (Malcolm 2023-08-29) not sure what --max-tokens does
            f"--max-tokens {TOKENS_PER_SAMPLE * args.batch_size}",
            # musicbert sets num workers to 0 for unknown reasons
            # TODO: (Malcolm 2023-08-29) test number of workers
            f"--num-workers {CPUS_ON_NODE}",
        ]
    ).split()
    + args_to_pass_on
)

print(" ".join(["fairseq-train"] + ARGS))

# Counterintuitively, the command name (`fairseq_train`) needs to be the first element
#   in the list of args the list of arguments
os.execvp("fairseq-train", ["fairseq-train"] + ARGS)
