#!/bin/bash
# Train Stage 1 MotionGPT3 with original diffusion (baseline)
cd /mnt/data8tb/Documents/repo/MotionGPT3
export WANDB_MODE=online
export PYTHONUNBUFFERED=1  # Disable Python output buffering
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# Limit CPU thread usage to prevent overload
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

source /mnt/data8tb/Documents/repo/MotionGPT3/.mgpt3/bin/activate
python -u train.py \
    --cfg configs/MoT_vae_stage1_t2m-flow.yaml \
    --cfg_assets configs/assets.yaml \
    --nodebug
