#!/bin/bash
#SBATCH --job-name=full_precision_asr_train
#SBATCH --gres=gpu:2           # Request 2 GPUs
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

source .venv/bin/activate

# ------------------------------------------------------------------
# RUN COMMAND
# Note: We use --nproc_per_node=2 because we are using 2 GPUs.
# ------------------------------------------------------------------
torchrun --nproc_per_node=2 -m onebit_asr.train_full_precision \
    --data_dir data \
    --epochs 50 \
    --train_data_fraction 0.1 \
    --batch_size 8 \
    --ddp