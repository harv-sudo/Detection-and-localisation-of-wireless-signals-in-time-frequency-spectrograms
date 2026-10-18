#!/bin/bash
#SBATCH --job-name=unet-f1-200
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00

set -euo pipefail

ROOT="/mnt/data2/users/$USER/wireless-signals"
PROJECT="$HOME/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms"

DATA_DIR="$ROOT/datasets/raw_large"
RUN_DIR="$ROOT/runs/unet_${SLURM_JOB_ID}"

module load python/3.11.9-gcc14.2.0
source "$ROOT/envs/main/bin/activate"

export PYTHONUNBUFFERED=1

cd "$PROJECT"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Training directory: $RUN_DIR"

nvidia-smi

# Make sure the dataset exists
if ! find "$DATA_DIR" -maxdepth 1 -name '*.mat' -type f \
    -print -quit | grep -q .; then
    echo "ERROR: Training dataset is empty"
    exit 1
fi

# Train U-Net
python -u UNet/experiment.py train \
    --model unet \
    --data-dir "$DATA_DIR" \
    --run-dir "$RUN_DIR" \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.001 \
    --patience 20 \
    --min-delta 0.0005 \
    --lr-patience 6 \
    --lr-factor 0.5 \
    --min-lr 0.00001 \
    --workers 2 \
    --threads 2 \
    --seed 42 \
    --device cuda

# Evaluate the BEST model on the held-out test set
python -u UNet/experiment.py evaluate \
    --run-dir "$RUN_DIR" \
    --checkpoint best \
    --workers 2 \
    --threads 2 \
    --device cuda

echo "Training and evaluation completed!"