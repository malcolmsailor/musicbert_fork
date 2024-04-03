import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
from fairseq.optim.lr_scheduler.cosine_lr_scheduler import (
    CosineLRSchedule,
    CosineLRScheduleConfig,
)
from fairseq.optim.lr_scheduler.inverse_square_root_schedule import (
    InverseSquareRootLRScheduleConfig,
    InverseSquareRootSchedule,
)
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import (
    PolynomialDecayLRSchedule,
    PolynomialDecayLRScheduleConfig,
)
from fairseq.optim.lr_scheduler.tri_stage_lr_scheduler import (
    TriStageLRSchedule,
    TriStageLRScheduleConfig,
)
from fairseq.optim.lr_scheduler.triangular_lr_scheduler import (
    TriangularLRSchedule,
    TriangularLRScheduleConfig,
)
from matplotlib import pyplot as plt

WARMUP_STEPS = 2500
TOTAL_STEPS = 50_000
STEPS_PER_UPDATE = 100

N_UPDATES = TOTAL_STEPS // STEPS_PER_UPDATE
INIT_LR = 0.00025


def get_lrs(lr_scheduler):
    lr_scheduler.step_begin_epoch(0)
    out = []
    for step_i in range(N_UPDATES + 1):
        lr_scheduler.optimizer.step()
        lr = lr_scheduler.step_update(step_i * STEPS_PER_UPDATE)
        out.append(lr)
    return np.array(out)


def make_plot(ax_name, data, ax):
    for name, ys in data.items():
        ax.plot(ys, label=name)
    ax.legend()
    ax.set_title(ax_name)


def main():
    model = nn.Linear(100, 100)
    # optim = torch.optim.Adam(model.parameters(), lr=0.05)

    optim = FairseqAdam(
        FairseqAdamConfig(lr=[INIT_LR]), model.parameters()  # type:ignore
    )

    default_config = PolynomialDecayLRScheduleConfig(
        total_num_update=TOTAL_STEPS, lr=[INIT_LR], warmup_updates=WARMUP_STEPS
    )
    default_scheduler = PolynomialDecayLRSchedule(default_config, optim)
    polynomial_config = PolynomialDecayLRScheduleConfig(
        total_num_update=TOTAL_STEPS,
        lr=[INIT_LR],
        power=2.0,
        warmup_updates=WARMUP_STEPS,
    )
    polynomial_scheduler = PolynomialDecayLRSchedule(polynomial_config, optim)

    inv_config = InverseSquareRootLRScheduleConfig(
        lr=[INIT_LR], warmup_updates=WARMUP_STEPS
    )
    inv_lr_scheduler = InverseSquareRootSchedule(inv_config, optim)

    tri_stage_config = TriStageLRScheduleConfig(
        # warmup_steps=WARMUP_STEPS,
        # hold_steps=WARMUP_STEPS,
        # decay_steps=TOTAL_STEPS - (2 * WARMUP_STEPS),
        lr=[INIT_LR],
        max_update=TOTAL_STEPS,
        phase_ratio=(0.05, 0.05, 0.9)
        # init_lr_scale=INIT_LR / 2,
        # final_lr_scale=INIT_LR / 10,
    )
    tri_stage_lr_scheduler = TriStageLRSchedule(tri_stage_config, optim)

    cosine_config = CosineLRScheduleConfig(
        lr=[INIT_LR], max_update=TOTAL_STEPS + 1, warmup_updates=WARMUP_STEPS
    )
    cosine_lr_scheduler = CosineLRSchedule(cosine_config, optim)
    cosine_3_cycle_config = CosineLRScheduleConfig(
        lr=[INIT_LR],
        lr_period_updates=math.ceil((TOTAL_STEPS - WARMUP_STEPS) / 3),
        lr_shrink=0.25,
        warmup_updates=WARMUP_STEPS,
    )
    cosine_3_cycle_scheduler = CosineLRSchedule(cosine_3_cycle_config, optim)
    # cosine_config3 = CosineLRScheduleConfig(
    #     lr=[INIT_LR], lr_period_updates=TOTAL_STEPS // 4, lr_shrink=0.5
    # )
    # cosine_lr_scheduler3 = CosineLRSchedule(cosine_config3, optim)
    # cosine_config4 = CosineLRScheduleConfig(
    #     lr=[INIT_LR], lr_period_updates=TOTAL_STEPS // 4, lr_shrink=0.25, t_mult=3.0
    # )
    # cosine_lr_scheduler4 = CosineLRSchedule(cosine_config4, optim)

    # triangular_config = TriangularLRScheduleConfig(
    #     lr=[INIT_LR / 2], max_lr=INIT_LR, lr_period_updates=10000, lr_shrink=0.1
    # )
    # triangular_lr_scheduler = TriangularLRSchedule(triangular_config, optim)
    # triangular_config2 = TriangularLRScheduleConfig(
    #     lr=[INIT_LR / 2],
    #     max_lr=INIT_LR,
    #     lr_period_updates=10000,
    #     lr_shrink=0.25,
    #     shrink_min=True,
    # )
    # triangular_lr_scheduler2 = TriangularLRSchedule(triangular_config2, optim)

    schedulers = {
        "default": default_scheduler,
        "polynomial, power=0.5": polynomial_scheduler,
        "cosine one-cycle": cosine_lr_scheduler,
        "cosine three-cycle": cosine_3_cycle_scheduler,
        "inv_sqrt": inv_lr_scheduler,
        "tri_stage": tri_stage_lr_scheduler,
    }
    #     "inverse_square_root": {
    #         "inverse_square_root": inv_lr_scheduler,
    #         "tri_stage1": tri_stage_lr_scheduler,
    #     },
    #     "cosine": {
    #         "with max update": cosine_lr_scheduler,
    #         "two periods, shrink=0.25": cosine_lr_scheduler2,
    #         "four periods, shrink=0.5": cosine_lr_scheduler3,
    #         "two periods, t_mult=3.0": cosine_lr_scheduler4,
    #     },
    #     "triangular": {
    #         "triangular1": triangular_lr_scheduler,
    #         "triangular2": triangular_lr_scheduler2,
    #     },
    # }
    results = {name: get_lrs(scheduler) for name, scheduler in schedulers.items()}
    # for group, contents in schedulers.items():
    #     results[group] = {
    #         name: get_lrs(scheduler) for name, scheduler in contents.items()
    #     }

    fig, ax = plt.subplots(figsize=(10, 10))
    make_plot("", results, ax)

    plt.show()


if __name__ == "__main__":
    main()
