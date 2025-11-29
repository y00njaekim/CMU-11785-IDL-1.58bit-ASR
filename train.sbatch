#!/bin/bash
#SBATCH --job-name=onebit_asr_train
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Activate virtual environment
source .venv/bin/activate

# Run training
torchrun --nproc_per_node=4 -m onebit_asr.train \
    --data_dir data \
    --epochs 20 \
    --batch_size 4

