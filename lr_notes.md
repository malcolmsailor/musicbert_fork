<!-- spellcheck-off -->

# Tristage learning rate scheduler

Implement the learning rate scheduler in https://arxiv.org/pdf/1904.08779.pdf

Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
three stages LR scheduling:

    - warmup stage, starting from `lr` * `init_lr_scale`, linearly
      increased to `lr` in `warmup_steps` iterations

    - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
      iterations

    - decay stage, after hold stage, decay LR exponetially to
      `lr` * `final_lr_scale` in `decay_steps`;
      after that LR is keep as `final_lr_scale` * `lr`

During warmup::

  init_lr = cfg.init_lr_scale * cfg.lr
  lrs = torch.linspace(init_lr, cfg.lr, cfg.warmup_steps)
  lr = lrs[update_num]

During hold::

  lr = cfg.lr

During decay::

  decay_factor = - math.log(cfg.final_lr_scale) / cfg.decay_steps
  lr = cfg.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)

After that::

  lr = cfg.lr * cfg.final_lr_scale

# Triangular lr scheduler

Assign LR based on a triangular cyclical schedule.

  See https://arxiv.org/pdf/1506.01186.pdf for details.

# Cosine LR Schedule

Assign LR based on a cyclical schedule that follows the cosine function.

See https://arxiv.org/pdf/1608.03983.pdf for details.

We also support a warmup phase where we linearly increase the learning rate
from some initial learning rate (``--warmup-init-lr``) until the configured
max learning rate (``--lr``).

During warmup::

  lrs = torch.linspace(cfg.warmup_init_lr, cfg.lr, cfg.warmup_updates)
  lr = lrs[update_num]

After warmup::

  lr = cfg.min_lr + 0.5*(cfg.lr - cfg.min_lr)*(1 + cos(t_curr / t_i))

where ``t_curr`` is current percentage of updates within the current period
range and ``t_i`` is the current period range, which is scaled by ``t_mul``
after every iteration.
