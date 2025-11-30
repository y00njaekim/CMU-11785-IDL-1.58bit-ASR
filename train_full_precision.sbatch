#!/bin/bash
#SBATCH --job-name=full_precision_asr_train
#SBATCH --gres=gpu:2           # Request 2 GPUs
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=train_full_precision_%j.out
#SBATCH --error=train_full_precision_%j.err

source .venv/bin/activate

# ------------------------------------------------------------------
# RUN COMMAND for Full Precision Training
# Note: We use --nproc_per_node=2 because we are using 2 GPUs.
# Using 1% of training data and 1% of validation data
# ------------------------------------------------------------------
torchrun --nproc_per_node=2 -m onebit_asr.train_full_precision \
    --data_dir data \
    --epochs 50 \
    --train_data_fraction 0.01 \
    --valid_data_fraction 0.01 \
    --batch_size 8 \
    --ddp

