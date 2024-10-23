cd /data3/zc/code/medical-sca/Medical-SAM2
export HF_HOME=/data3/zc/cache/huggingface
CUDA_LAUNCH_BLOCKING=1 \
deepspeed train_3d_sca_ddp.py -deepspeed deepspeed/zero2.json \
    -batch_size 1 \
    -net sca2 \
    -exp_name M3D_MedSCA2 \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sca2_hiera_s_llama3_1b \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 1 \
    -dataset m3d_cap \
    -data_path ../../../data/M3D-Cap \
    -video_length 2 \
    -cap_data_path ../../../data/M3D-Cap/M3D_Cap/M3D_Cap.json 
