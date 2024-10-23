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

export HF_HOME=/data3/zc/cache/huggingface
CUDA_LAUNCH_BLOCKING=1 python eval_3d_sca_ddp.py \
    -net sca2 \
    -exp_name M3D_MedSAM2 \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sca2_hiera_s \
    -language_model_name_or_path microsoft/phi-1_5 \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 4 \
    -dataset m3d_cap \
    -data_path ../../../data/M3D-Cap \
    -video_length 4 \
    -cap_data_path ../../../data/M3D-Cap/M3D_Cap/M3D_Cap.json \
    -weight_path /data3/zc/code/medical-sca/Medical-SAM2/logs/M3D_MedSAM2_2024_10_19_21_16_58/Model/step-10000.pth
