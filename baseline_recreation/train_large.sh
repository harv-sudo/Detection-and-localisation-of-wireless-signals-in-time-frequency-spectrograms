#!/bin/bash
#SBATCH --job-name=sae_large
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sae_large_%j.out
#SBATCH --error=logs/sae_large_%j.err

cd /users/sghkulka/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation

mkdir -p logs checkpoints

source venv/bin/activate

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -u train.py