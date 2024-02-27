#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=linear_probe
#SBATCH --time=012:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /home/ms3682/slurm_output/%j.out # Note that directory will not be created if it does not exist; also, ~ expansion doesn't seem to work


if [[ -z $1 ]];
then
    echo Error, sweep command must be provided as first argument
    exit 1
fi

sweep_cmd=$1
count=${2:-10}

module load miniconda
conda activate newbert

set -x
# Getting dirname in this way doesn't work within slurm
# cd $(dirname "$0")
cd ~/code/musicbert_fork/experiments/linear_probe

## For unknown reasons, getting the sweep id programmatically like this is not working.
## Therefore, we'll need to launch the sweep separately with
##      wandb sweep --project [project name] linear_probe_config.yaml
## Then copy the sweep command (resembles msailor/sweep-test/lvvke2n5) and pass it as 
##  an arg to this script
# export sweep_id=$(wandb sweep --project sweep-test linear_probe_config.yaml | sed 's/\x1b\[[0-9;]*m//g' | awk '/Creating sweep with ID:/{print $NF}')
# echo $sweep_id
# sleep 5 # I'm not sure sleeping is necessary but it seems prudent

# Concerning these env vars see https://github.com/wandb/wandb/issues/2420#issuecomment-888480380
export WANDB_CONFIG_DIR=/tmp/
export WANDB_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/

wandb agent --count $count $sweep_cmd



set +x
