import argparse
import os
import re
import shutil
import subprocess
import sys


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

PEAK_LR = 0.01

BATCH_SIZE = 64
MAX_SENTENCES = 4

TOKENS_PER_SAMPLE = 8192

HEAD_NAME = "sequence_tagging_head"

UPDATE_FREQ_DENOM = max(N_GPU_LOCAL, 1)
UPDATE_FREQ = BATCH_SIZE // (MAX_SENTENCES * UPDATE_FREQ_DENOM)


SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
USER_DIR = os.path.join(SCRIPT_DIR, "../musicbert")


parser = argparse.ArgumentParser()
parser.add_argument("--data-bin-dir", "-d", required=True)
parser.add_argument("--architecture", "-a", required=True)
parser.add_argument("--wandb-project", "-W", required=True)
parser.add_argument("--total-updates", "-u", type=int, default=TOTAL_UPDATES)
parser.add_argument("--warmup-updates", "-w", type=int, default=WARMUP_UPDATES)
parser.add_argument("--checkpoint", "-c")
args, args_to_pass_on = parser.parse_known_args()

DATA_BIN_DIR = args.data_bin_dir
NN_ARCH = f"musicbert_{args.architecture}"
WANDB_PROJECT = args.wandb_project

RESTORE_FLAG = "" if not args.checkpoint else f"--restore-file {args.checkpoint}"

label_dict_file = os.path.join(DATA_BIN_DIR, "label", "dict.txt")
with open(label_dict_file, "r") as label_file:
    NUM_CLASSES = len(
        [line for line in label_file if not re.match(r"madeupword[0-9]{4}", line)]
    )

ARGS = (
    " ".join(
        [
            DATA_BIN_DIR,
            CPU_FLAG,
            f"--user-dir {USER_DIR}",
            RESTORE_FLAG,
            f"--wandb-project {WANDB_PROJECT}",
            "--task musicbert_sequence_tagging",
            f"--arch {NN_ARCH}",
            f"--batch-size {MAX_SENTENCES}",
            f"--update-freq {UPDATE_FREQ}",
            "--criterion sequence_tagging",
            f"--classification-head-name {HEAD_NAME}",
            "--compound-token-ratio 8",
            f"--num-classes {NUM_CLASSES}",
            "--reset-optimizer",
            "--reset-dataloader",
            "--reset-meters",
            "--optimizer adam",
            "--adam-betas (0.9,0.98)",
            "--adam-eps 1e-6",
            "--clip-norm 0.0",
            "--lr-scheduler polynomial_decay",
            f"--lr {PEAK_LR}",
            "--log-format simple",
            f"--warmup-updates {args.warmup_updates}",
            f"--total-num-update {args.total_updates}",
            f"--max-update {args.total_updates}",
            "--no-epoch-checkpoints",
            "--find-unused-parameters",
            "--best-checkpoint-metric accuracy",
            "--maximize-best-checkpoint-metric",
            "--max-positions 8192",
            "--required-batch-size-multiple 1",
            f"--max-tokens {TOKENS_PER_SAMPLE * MAX_SENTENCES}",
        ]
    ).split()
    + args_to_pass_on
)

print(" ".join(["fairseq-train"] + ARGS))

# Counterintuitively, the command name (`fairseq_train`) needs to be the first element
#   in the list of args the list of arguments
os.execvp("fairseq-train", ["fairseq-train"] + ARGS)
