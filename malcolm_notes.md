# Environments

`musicbert`: using version of fairseq specified on Github from `git+https://github.com/pytorch/fairseq@336942734c85791a90baa373c212d27e7c722662#egg=fairseq`

`musicbert2`: another attempt

`musicbert3`: yet another attempt, installing numpy=1.23 downgrades torch to CPU version (from conda-forge channel), tried installing numpy then re-installing torch (from torch channel) 07:57

`musicbert4`: 07:57 Trying to install numpy with pip. As of 2023-08-28 08:32 this environment seems to work! 

`newbert`: an attempt to recreate the `musicbert4` environment

# Error messages

2023-08-28 I was getting this message:
```
/gpfs/gibbs/project/quinn/ms3682/conda_envs/musicbert2/lib/python3.8/site-packages/torch/cuda/__init__.py:546: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
```
It seems to be because I was debugging on CPU-only. On a GPU partition it appears to go away.

2023-08-28 Running fairseq I get the warning below. Judging by the Pytorch thread here `https://discuss.pytorch.org/t/amp-c-fused-kernels-unavailable/136071` this is probably not a priority.
```
UserWarning: amp_C fused kernels unavailable, disabling multi_tensor_l2norm; you may get better perfo
rmance by installing NVIDIA's apex library
```

# 2023-08-28

Trying to run `bash train_genre.sh ~/project/datasets/topmagd 13 0 ~/project/checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt` and having environment problems.

07:47: run on CPU was killed for unknown reason; hoping it'll work on GPU... nope :(

08:24: running `bash train_mask.sh -a small -d ~/project/datasets/octmidi_data_bin -r musicbert` on interactive GPU to make sure
- [y] env runs
- [y] GPU is used

08:28: running `bash train_genre.sh ~/project/datasets/topmagd 13 0 ~/project/checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt` to see if I can finetune from provided checkpoint
- [y] env runs
- [y] GPU is used

## Recreating environment

1. Run `conda create --name [new env name] --file [musicbert4.txt]` where `musicbert4.txt` was created with `conda list --explicit > musicbert4.txt`

2. `pip freeze ...` and then `pip -r ...` for the remaining requirements doesn't seem to be working. So instead it looks like we need to run:
    - `pip install git+https://github.com/pytorch/fairseq@336942734c85791a90baa373c212d27e7c722662#egg=fairseq`
    - `pip install numpy==1.23` (We require 1.23 because < 1.23 gives Cython error and > 1.23 gives error because np.float and np.int have been removed (they are used by fairseq) and it doesn't seem to be possible to monkeypatch them before fairseq tries to access them.)
    - `pip install scikit-learn==1.3.0` (Not sure we need to freeze version)

## wandb

Wandb login is cached (presumably key is stored in config somewhere).
Use `--wandb-project ${WANDB_PROJECT}` with fairseq to log.
