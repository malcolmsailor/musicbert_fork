# Environments

`musicbert`: using version of fairseq specified on Github from `git+https://github.com/pytorch/fairseq@336942734c85791a90baa373c212d27e7c722662#egg=fairseq`

`musicbert2`: another attempt

`musicbert3`: yet another attempt, installing numpy=1.23 downgrades torch to CPU version (from conda-forge channel), tried installing numpy then re-installing torch (from torch channel) 07:57

`musicbert4`: 07:57 Trying to install numpy with pip. As of 2023-08-28 08:32 this environment seems to work! 
- NB This env is missing:
    - wandb
    - miditoolkit (required for preprocessing w/ musicbert)
    - matplotlib (unspecified requirement for miditoolkit)
    - I have installed each of these (with pip) in the `newbert` env; I don't believe that these installations cause any issues.

`newbert`: an attempt to recreate the `musicbert4` environment
- I have actually been using this environment

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

- Wandb login is cached (presumably key is stored in config somewhere). Use `--wandb-project ${WANDB_PROJECT}` with fairseq to log.

- Wandb run name is initialized in fairseq-train like this:
```python
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
```
    This is kind of annoying as we don't take advantage of the mnemonic random naming of runs provided by wandb. Perhaps if WANDB_NAME is "" we will get a random name?

- Wandb says 1 GPU even when there is more than 1 GPU. This seems to be incorrect---nvidia-smi on the running job lists multiple gpus in use.


# 2023-08-29

Finally got classification to work by basing it more closely on https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.custom_classification.md. It turns out not to be necessary to write a custom task. (Or, at least, so it appears---I'll need to actually see the results.)

Next task is to see if I can overfit.

To create a subset for overfitting,
- 1. run `MUSICBERT_COMPOSER_LIMIT=<LIMIT> MUSICBERT_DATA_LIMIT=<LIMIT> python preprocess_composer_classification.py` substituting an appropriate file limit. (The files are cut into multiple segments so there will be more samples than files.)
  - TODO maybe we want to look at sampling randomly from each file rather than sampling deterministically?
- 2. [Optional] Run `bash repeat_train_set.sh <path> <n repetitions>` so that the training set is longer.
- 2. run `bash binarize_composer_classification.sh /Users/malcolm/datasets/composer_classification_limit_2_data_raw` (adjusting the path as necessary)

Training:

2023-08-29 13:43 overfitting on 5 files with 2 composers seems to work
`bash train_composer_classification.sh -d ~/project/datasets/composer_classification_file_limit_5_composer_limit_2_data_bin -r musicbert/ -a base -W composer_classification_overfit -c ~/project/checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt -w 5 -l 2`
That said, loss goes to 0 so fast I'm wondering if there isn't a bug. It could also be that there are other obvious differences between the files (e.g., of orchestration or key) I suppose.

# 2023-08-30

Training:
- finetuning composer classification:
    - this seems to work, although by the first validation epoch we're already ~82% acc and we seem to mostly be overfitting (~97% train acc) [https://wandb.ai/msailor/composer_classification/runs/zoqbk55s?workspace=user-msailor](https://wandb.ai/msailor/composer_classification/runs/zoqbk55s?workspace=user-msailor)
    - freezing the encoder and training only the classification head gives much worse results but doesn't overfit as quickly: [https://wandb.ai/msailor/composer_classification/runs/f5eld9f8?workspace=user-msailor](https://wandb.ai/msailor/composer_classification/runs/f5eld9f8?workspace=user-msailor)


