#!/bin/bash
# TODO(neil): partition, account, GPU type, conda env, data path — none confirmed.
#SBATCH --job-name=ekfac_fit
#SBATCH --partition=TODO_neil_partition
#SBATCH --account=TODO_neil_account
#SBATCH --gres=gpu:TODO_neil_gpu_type:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ekfac_fit_%j.out

set -euo pipefail

# TODO(neil): conda env name
source activate TODO_neil_conda_env

cd "$SLURM_SUBMIT_DIR"

# TODO(neil): confirm checkpoint path and data root on Talapas
python -m experiments.layer_scores.ekfac_fit \
    --checkpoint TODO_neil_checkpoint_path \
    --data-root TODO_neil_data_path \
    --out results/layer_scores/${SLURM_JOB_ID}/factors.h5 \
    --device cuda \
    --n-fac 50000 \
    --n-corr 10000
