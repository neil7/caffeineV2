#!/bin/bash
# TODO(neil): partition, account, GPU type, conda env, data path — none confirmed.
# Single-GPU, full pipeline: queries -> scores -> E1 -> E2 -> E3.
# Caches factors.h5/queries.json/scores.parquet so a re-submitted job resumes
# instead of recomputing steps whose output files already exist.
#SBATCH --job-name=layer_scores
#SBATCH --partition=TODO_neil_partition
#SBATCH --account=TODO_neil_account
#SBATCH --gres=gpu:TODO_neil_gpu_type:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/layer_scores_%j.out

set -euo pipefail
source activate TODO_neil_conda_env
cd "$SLURM_SUBMIT_DIR"

RUN_DIR=results/layer_scores/${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
CKPT=TODO_neil_checkpoint_path
DATA_ROOT=TODO_neil_data_path

[ -f "$RUN_DIR/factors.h5" ] || python -m experiments.layer_scores.ekfac_fit \
    --checkpoint "$CKPT" --data-root "$DATA_ROOT" --out "$RUN_DIR/factors.h5" --device cuda

[ -f "$RUN_DIR/queries.json" ] || python -m experiments.layer_scores.queries \
    --checkpoint "$CKPT" --data-root "$DATA_ROOT" --out "$RUN_DIR/queries.json" --device cuda

[ -f "$RUN_DIR/scores.parquet" ] || python -m experiments.layer_scores.scores \
    --checkpoint "$CKPT" --factors-h5 "$RUN_DIR/factors.h5" --queries "$RUN_DIR/queries.json" \
    --data-root "$DATA_ROOT" --out "$RUN_DIR/scores.parquet" --device cuda

python -m experiments.layer_scores.e1_concentration --scores "$RUN_DIR/scores.parquet" --out-dir "$RUN_DIR/e1"
python -m experiments.layer_scores.e2_variation --scores "$RUN_DIR/scores.parquet" --out-dir "$RUN_DIR/e2"
python -m experiments.layer_scores.e3_bounds \
    --checkpoint "$CKPT" --factors-h5 "$RUN_DIR/factors.h5" --queries "$RUN_DIR/queries.json" \
    --scores "$RUN_DIR/scores.parquet" --data-root "$DATA_ROOT" --out-dir "$RUN_DIR/e3" --device cuda
