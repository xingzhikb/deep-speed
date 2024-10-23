python -m debugpy --wait-for-client --listen 0.0.0.0:5678 \
    -m src.train \
    train_data='[m3d_2d]' eval_data='[m3d_2d]' \
    +model=base_sca_multitask_v2 \
    # model.cache_dir=.model.cache/ \
    training.do_train=True \
    training.do_eval=True \
    training.fp16=True \
    training.num_masks_per_sample=16 \
    training.per_device_train_batch_size=1 \
    training.dataloader_num_workers=4 \
    training.max_steps=99 \
    training.logging_first_step=True \
    training.logging_steps=5 \
    training.evaluate_before_train=True \
    training.max_eval_samples=3 \
    training.eval_steps=50 \
    training.save_steps=50 \
    wandb.log=True \
    wandb.project='IU_xray' \
    training.dataloader_num_workers=4 \
    wandb.name='ft' \
    model.num_caption_tokens=8 \
    model.additional_num_hidden_layers=12 \
    model.num_task_tokens=6 \
    training.lr_scheduler_type=cosine \
    +data_transforms=lsj-0_1-2_0 \
    model.lm_head_model_name_or_path=gpt2 \
    model.sam_model_name_or_path=facebook/sam-vit-base



