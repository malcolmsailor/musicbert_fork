"""
Monkeypatches `fairseq.trainer.Trainer.load_checkpoint` so that it calls 
`load_state_dict()` with strict=False. This is needed to load pretrained checkpoints 
to multi-task token-classification model.
"""

import time

from fairseq import checkpoint_utils, distributed_utils, utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.trainer import Trainer, logger


def load_checkpoint(
    self,
    filename,
    reset_optimizer=False,
    reset_lr_scheduler=False,
    optimizer_overrides=None,
    reset_meters=False,
):
    """
    Load all training state from a checkpoint file.
    rank = 0 will load the checkpoint, and then broadcast it to all
    other ranks.
    """
    extra_state, self._optim_history, last_optim_state = None, [], None

    logger.info(f"Preparing to load checkpoint {filename}")
    is_distributed = self.data_parallel_world_size > 1
    bexists = PathManager.isfile(filename)
    if bexists:
        load_on_all_ranks = (
            self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
            # TPUs don't support broadcast yet, so load checkpoints
            # on every worker for now
            or self.tpu
        )

        if load_on_all_ranks or self.data_parallel_rank == 0:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                filename, load_on_all_ranks=load_on_all_ranks
            )
            last_optim_state = state.get("last_optimizer_state", None)

            # If doing zero_sharding, do not broadcast global optimizer
            # state. Later we will broadcast sharded states to each rank
            # to avoid memory from exploding.
            if (
                not load_on_all_ranks
                and self.cfg.distributed_training.zero_sharding == "os"
                and "last_optimizer_state" in state
                and is_distributed
            ):
                state["last_optimizer_state"] = "SHARDED"
        else:
            last_optim_state = None
            state = None

        if is_distributed and not load_on_all_ranks:
            state = distributed_utils.broadcast_object(
                state,
                src_rank=0,
                group=self.data_parallel_process_group,
                dist_device=self.device,
            )
            if self.data_parallel_rank > 0:
                last_optim_state = state.get("last_optimizer_state", None)

        # load model parameters
        try:
            self.get_model().load_state_dict(
                state["model"], strict=False, model_cfg=self.cfg.model
            )
            if utils.has_parameters(self.get_criterion()):
                self.get_criterion().load_state_dict(state["criterion"], strict=True)
        except Exception:
            raise Exception(
                "Cannot load model parameters from checkpoint {}; "
                "please ensure that the architectures match.".format(filename)
            )
        extra_state = state["extra_state"]
        self._optim_history = state["optimizer_history"]

    if last_optim_state is not None and not reset_optimizer:
        # rebuild optimizer after loading model, since params may have changed
        self._build_optimizer()

        # only reload optimizer and lr_scheduler if they match
        last_optim = self._optim_history[-1]
        assert (
            last_optim["criterion_name"] == self.get_criterion().__class__.__name__
        ), "Criterion does not match; please reset the optimizer (--reset-optimizer)."
        assert (
            last_optim["optimizer_name"] == self.optimizer.__class__.__name__
        ), "Optimizer does not match; please reset the optimizer (--reset-optimizer)."

        if not reset_lr_scheduler:
            self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

        if not load_on_all_ranks and is_distributed:
            last_optim_state = self.optimizer.broadcast_global_state_dict(
                last_optim_state
            )
        self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

        self.set_num_updates(last_optim["num_updates"])

    if extra_state is not None:
        epoch = extra_state["train_iterator"]["epoch"]

        if "previous_training_time" in extra_state:
            self._previous_training_time = extra_state["previous_training_time"]
            self._start_time = time.time()

        self.lr_step(epoch)

        if "metrics" in extra_state and not reset_meters:
            metrics.load_state_dict(extra_state["metrics"])

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in metrics.get_meters("default"):
                if isinstance(meter, meters.TimeMeter):
                    meter.reset()

        logger.info(
            "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                filename, epoch, self.get_num_updates()
            )
        )

    else:
        logger.info("No existing checkpoint found {}".format(filename))

    return extra_state


Trainer.load_checkpoint = load_checkpoint
