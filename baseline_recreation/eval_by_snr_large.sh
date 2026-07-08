#!/bin/bash
#SBATCH --job-name=sae_snr_eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sae_snr_eval_%j.out
#SBATCH --error=logs/sae_snr_eval_%j.err

cd /users/sghkulka/Detection-and-localisation-of-wireless-signals-in-time-frequency-spectrograms/baseline_recreation

source venv/bin/activate

python -u evaluate_by_snr.py