#!/bin/bash
#SBATCH --job-name=quantized_asr_train
#SBATCH --gres=gpu:2           # Request 2 GPUs
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=train_quantized_%j.out
#SBATCH --error=train_quantized_%j.err
#SBATCH --nodelist=catalyst-0-11  # Specify node to use catalyst-0-11

# Set environment variables for stability
export PYTHONUNBUFFERED=1
export SPM_NUM_THREADS=1

source .venv/bin/activate

# ------------------------------------------------------------------
# RUN COMMAND for Quantized Training (1-bit/2-bit)
# Note: We use --nproc_per_node=2 because we are using 2 GPUs.
# Using 100% of training data and 100% of validation data
# ------------------------------------------------------------------
torchrun --nproc_per_node=2 -m onebit_asr.train \
    --data_dir data \
    --epochs 50 \
    --train_data_fraction 0.01 \
    --valid_data_fraction 1.0 \
    --batch_size 8 \
    --ddp

