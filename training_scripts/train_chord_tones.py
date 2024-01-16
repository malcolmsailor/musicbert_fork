import argparse
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from itertools import count

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def shell(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True).stdout.decode()


# For the sake of reproducibility, we want to commit all changes in the directory before
# running. Then we can get the git hash and command from wandb to reproduce (hopefully!)

if not os.getenv("DEBUG_MUSICBERT", None):
    uncommited_changes = shell("git status --porcelain")
    if uncommited_changes:
        print("There are uncommitted changes; commit them then rerun (or set DEBUG_MUSICBERT env variable)")
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
        "musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt",
    ),
)

PEAK_LR = 0.0005  # Borrowed from musicbert

# NB in musicbert scripts, UPDATE_BATCH_SIZE is only used in the UPDATE_FREQ calculation below;
#   the actual batch size to fairseq-train is set by BATCH_SIZE arg
UPDATE_BATCH_SIZE = 64
BATCH_SIZE = 4


SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))
USER_DIR = os.path.join(SCRIPT_DIR, "../musicbert")

parser = argparse.ArgumentParser()
parser.add_argument("--data-bin-dir", "-d", required=True)
parser.add_argument("--architecture", "-a")
parser.add_argument("--wandb-project", "-W")
parser.add_argument("--total-updates", "-u", type=int, default=TOTAL_UPDATES)
parser.add_argument("--warmup-updates", "-w", type=int, default=WARMUP_UPDATES)
parser.add_argument("--update-batch-size", type=int, default=UPDATE_BATCH_SIZE)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--lr", type=float, default=PEAK_LR)
parser.add_argument("--lr-scheduler", type=str, default="polynomial_decay")
parser.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT)
parser.add_argument("--multitarget", action="store_true")
parser.add_argument("--sequence-level", action="store_true")
parser.add_argument("--dryrun", action="store_true")
parser.add_argument("--skip-training", action="store_true")
parser.add_argument("--skip-test-metrics", action="store_true")
parser.add_argument("--skip-predict", action="store_true")
parser.add_argument(
    "--predict-splits", nargs="+", default=["test"], choices=["test", "train", "valid"]
)
parser.add_argument("--predict-max-examples", default=None, type=int)
parser.add_argument(
    "--run-name", type=str, help="required if skip-training, otherwise ignored"
)
parser.add_argument("--num-sample-inputs", type=int, default=4)
parser.add_argument(
    "--predict-args",
    type=str,
    help="args to pass through unchanged to the prediction script, encapsulated in a single string",
    default=None,
)
args, args_to_pass_on_to_train = parser.parse_known_args()

assert not (args.sequence_level and args.multitarget)

TOKENS_PER_SAMPLE = 8192

UPDATE_FREQ_DENOM = max(N_GPU_LOCAL, 1)
UPDATE_FREQ = min(args.update_batch_size // (args.batch_size * UPDATE_FREQ_DENOM), 1)

NEW_CHECKPOINTS_DIR = os.environ["NEW_CHECKPOINTS_DIR"]
PREDICTIONS_DIR = os.environ["SAVED_PREDICTIONS_DIR"]
EXAMPLE_INPUTS_DIR = os.environ["EXAMPLE_INPUTS_DIR"]

SLURM_ID = os.getenv("SLURM_JOB_ID", None)

if SLURM_ID is not None:
    # We're running a Slurm job
    SAVE_DIR = os.path.join(NEW_CHECKPOINTS_DIR, "musicbert_fork", SLURM_ID)
    # We can't use os.cpu_count() because it will show all the cpus on the node
    #   rather than just those allocated to our job
    CPUS_ON_NODE = os.getenv("SLURM_CPUS_ON_NODE", 1)
    PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, "musicbert_fork", SLURM_ID)
    EXAMPLE_INPUTS_PATH = os.path.join(EXAMPLE_INPUTS_DIR, "musicbert_fork", SLURM_ID)
else:
    # We're not running a Slurm job
    SAVE_DIR = os.path.join(
        NEW_CHECKPOINTS_DIR, "musicbert_fork", str(round(time.time()))
    )
    PREDICTIONS_PATH = os.path.join(
        PREDICTIONS_DIR, "musicbert_fork", str(round(time.time()))
    )
    EXAMPLE_INPUTS_PATH = os.path.join(
        EXAMPLE_INPUTS_DIR, "musicbert_fork", str(round(time.time()))
    )

    CPUS_ON_NODE = os.cpu_count()

DATA_BIN_DIR = args.data_bin_dir.rstrip(os.path.sep)

if args.skip_training and args.skip_test_metrics:
    LOGGER.info("found --skip-training flag, skipping training")
    LOGGER.info("found --skip-test-metrics flag, skipping test metrics")
else:
    missing_args = []
    for arg, arg_name in [
        (args.architecture, "--architecture"),
        (args.wandb_project, "--wandb-project"),
    ]:
        if arg is None:
            missing_args.append(arg_name)
    if missing_args:
        raise ValueError(f"CLI args {missing_args} are required if training")

    if args.run_name and not args.skip_training:
        LOGGER.warning(f"--run-name is ignored if not skipping training")

    if not args.dryrun:
        os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)

    NN_ARCH = f"musicbert_{args.architecture}"
    WANDB_PROJECT = args.wandb_project
    WANDB_FLAG = (
        "" if WANDB_PROJECT == "scratch" else f"--wandb-project {WANDB_PROJECT}"
    )

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
                [
                    line
                    for line in label_file
                    if not re.match(r"madeupword[0-9]{4}", line)
                ]
            )

    if args.multitarget:
        TASK = "musicbert_multitarget_sequence_tagging"
        HEAD_NAME = "sequence_multitarget_tagging_head"
        CRITERION = "multitarget_sequence_tagging"
    elif args.sequence_level:
        TASK = "sentence_prediction"
        HEAD_NAME = "sequence_level_head"
        CRITERION = "freezable_sentence_prediction"
    else:
        TASK = "musicbert_sequence_tagging"
        HEAD_NAME = "sequence_tagging_head"
        CRITERION = "sequence_tagging"
    # TASK = (
    #     "musicbert_multitarget_sequence_tagging"
    #     if args.multitarget
    #     else "musicbert_sequence_tagging"
    # )

    # HEAD_NAME = (
    #     "sequence_multitarget_tagging_head"
    #     if args.multitarget
    #     else "sequence_tagging_head"
    # )
    # CRITERION = (
    #     "multitarget_sequence_tagging" if args.multitarget else "sequence_tagging"
    # )

    SHARED_ARGS = (
        " ".join(
            [
                DATA_BIN_DIR,
                CPU_FLAG,
                f"--user-dir {USER_DIR}",
                WANDB_FLAG,
                f"--task {TASK}",
                f"--arch {NN_ARCH}",
                f"--batch-size {args.batch_size}",
                f"--update-freq {UPDATE_FREQ}",
                f"--criterion {CRITERION}",
                f"--classification-head-name {HEAD_NAME}",
                "--compound-token-ratio 8" if not args.sequence_level else "",
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
                "--log-format simple",
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
                f"--lr-scheduler {args.lr_scheduler}",
                f"--lr {args.lr}",
                f"--example-network-inputs-to-save {args.num_sample_inputs}" if not args.sequence_level else "",
                f"--example-network-inputs-path {EXAMPLE_INPUTS_PATH}" if not args.sequence_level else "",
            ]
        ).split()
        + args_to_pass_on_to_train
    )

    TRAIN_ARGS = (
        SHARED_ARGS
        + " ".join(
            [
                RESTORE_FLAG,
                f"--save-dir {SAVE_DIR}",
                # tri_stage doesn't support warmup updates
                (
                    f"--warmup-updates {args.warmup_updates} "
                    if args.lr_scheduler != "tri_stage"
                    else ""
                )
                +
                # (Malcolm 2023-10-26) --total-num-update
                #   is only used by the polynomial decay lr scheduler
                (
                    f"--total-num-update {args.total_updates} "
                    if args.lr_scheduler == "polynomial_decay"
                    else ""
                )
                + f"--max-update {args.total_updates}",
                "--no-epoch-checkpoints",
            ]
        ).split()
    )
    if args.skip_training:
        SAVE_DIR = os.path.join(NEW_CHECKPOINTS_DIR, "musicbert_fork", args.run_name)
    BEST_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_best.pt")
    TEST_ARGS = (
        SHARED_ARGS
        + " ".join(
            [
                f"--restore-file {BEST_CHECKPOINT_PATH}",
                f"--save-dir {SAVE_DIR}",
                "--valid-subset test",
                # I'm not entirely sure why, but --load-checkpoint-heads seems to
                #   be necessary here but not elsewhere
                "--load-checkpoint-heads",
                "--max-epoch 0",
                # For whatever reason, --no-save doesn't seem to actually prevent
                #   saving checkpoints, so we also provide the other flags
                "--no-save",
                "--no-last-checkpoints",
                "--no-epoch-checkpoints",
                "--log-interval 128",
                (
                    f"--total-num-update 1 "
                    if args.lr_scheduler == "polynomial_decay"
                    else ""
                )
                + "--max-update 1",
            ]
        ).split()
    )

    if args.skip_training:
        LOGGER.info("found --skip-training flag, skipping training")
    else:
        LOGGER.info(
            " ".join(["fairseq-train"] + [shlex.quote(arg) for arg in TRAIN_ARGS])
        )
        if not args.dryrun:
            # (Malcolm 2023-10-27) There must have been some reason why I was using
            #   "execvp" instead of subprocess originally but I'm no longer sure
            #   what it was and, importantly, it overrides this process which means
            #   that exceution doesn't continue.
            # Counterintuitively, the command name (`fairseq_train`) needs to be the first element
            #   in the the list of arguments
            # os.execvp("fairseq-train", ["fairseq-train"] + TRAIN_ARGS)
            subprocess.run(["fairseq-train"] + TRAIN_ARGS, check=True)

    if args.skip_test_metrics:
        LOGGER.info("found --skip-test-metrics flag, skipping test metrics")
    else:
        LOGGER.info(
            " ".join(["fairseq-train"] + [shlex.quote(arg) for arg in TEST_ARGS])
        )
        if not args.dryrun:
            # TODO: (Malcolm 2023-10-31) try using fairseq-validate here
            # Counterintuitively, the command name (`fairseq_train`) needs to be the first element
            #   in the the list of arguments
            # os.execvp("fairseq-train", ["fairseq-train"] + TEST_ARGS)
            subprocess.run(["fairseq-train"] + TEST_ARGS, check=True)


if args.skip_predict:
    LOGGER.info("found --skip-predict flag, skipping prediction")
else:
    if args.skip_training:
        if not args.run_name:
            raise ValueError(f"--run-name is required if --skip-training")
        SAVE_DIR = os.path.join(NEW_CHECKPOINTS_DIR, "musicbert_fork", args.run_name)
        BEST_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_best.pt")
        PREDICTIONS_PATH = os.path.join(
            PREDICTIONS_DIR, "musicbert_fork", args.run_name
        )
    else:
        BEST_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_best.pt")

    if os.path.exists(BEST_CHECKPOINT_PATH) or args.dryrun:
        if args.predict_args is None:
            args_to_pass_on_to_predict = []
        else:
            args_to_pass_on_to_predict = args.predict_args.split()

        assert all(split in {"test", "valid", "train"} for split in args.predict_splits)

        for predict_split in args.predict_splits:
            # PREDICTIONS_OUTPUT = os.path.join(PREDICTIONS_PATH, predict_split)
            if args.multitarget:
                PREDICTIONS_SCRIPT = os.path.join(
                    SCRIPT_DIR, "..", "eval_scripts", "save_multi_target_predictions.py"
                )
            else:
                PREDICTIONS_SCRIPT = os.path.join(
                    SCRIPT_DIR, "..", "eval_scripts", "save_predictions.py"
                )
            if args.predict_max_examples is not None:
                max_example_str = f"--max-examples {args.predict_max_examples}"
            else:
                max_example_str = ""
            PREDICT_ARGS = (
                " ".join(
                    [
                        PREDICTIONS_SCRIPT,
                        f"--dataset {predict_split}",
                        f"--data-dir {DATA_BIN_DIR}",
                        f"--checkpoint {BEST_CHECKPOINT_PATH}",
                        f"--output-folder {PREDICTIONS_PATH}",
                        max_example_str,
                    ]
                ).split()
                + args_to_pass_on_to_predict
            )
            LOGGER.info(
                " ".join(["python"] + [shlex.quote(arg) for arg in PREDICT_ARGS])
            )
            if not args.dryrun:
                # os.execvp("python", ["python"] + PREDICT_ARGS)
                subprocess.run(["python"] + PREDICT_ARGS, check=True)

            # Copy the metadata file into the predictions folder too
            assert DATA_BIN_DIR.endswith("_bin")
            DATA_RAW_DIR = DATA_BIN_DIR[:-4] + "_raw"
            shutil.copy(
                os.path.join(DATA_RAW_DIR, f"metadata_{predict_split}.txt"),
                os.path.join(PREDICTIONS_PATH, f"metadata_{predict_split}.txt"),
            )
    else:
        LOGGER.info(f"Didn't find {BEST_CHECKPOINT_PATH}")
        raise FileNotFoundError(BEST_CHECKPOINT_PATH)
