#!/bin/bash
#SBATCH --job-name=med_sam_debug
#SBATCH --partition=hpg-ai
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=czhou94@ucsc.edu
#SBATCH --output=slurm/med_sam2_debug/%j_med_sam2_debug.slurm

export CUDA_VISIBLE_DEVICES=0
python train_3d.py \
    -net sam2 \
    -exp_name BTCV_MedSAM2 \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -video_length 2 \
    -dataset btcv \
    -data_path ../../../data/btcv \
