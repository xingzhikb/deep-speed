python train_3d_sca.py \
    -net sca2 \
    -exp_name M3D_MedSAM2 \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -sam_config sca2_hiera_t \
    -image_size 1024 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -dataset m3d \
    -tag "0005" \
    -multimask_output 3 \
    -data_path ../../../data/M3D-Seg/M3D_Seg_npy_old \
