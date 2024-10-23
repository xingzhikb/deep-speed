#!/bin/bash

# Check if the wandb name argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <wandb_name>"
    exit 1
fi

WANDB_NAME="$1"

# Generate timestamp
TIMESTAMP=$(python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))")

# Create directories
mkdir -p ./exp/checkpoints/$TIMESTAMP
mkdir -p ./exp/logs/$TIMESTAMP

HYDRA_FULL_ERROR=1 python -m src.train \
    train_data='[m3d_2d]' eval_data='[m3d_2d]' \
    +model=base_sca_multitask_v2 \
    training.do_train=True \
    training.do_eval=True \
    training.do_inference=True \
    +data.streaming=False \
    training.max_eval_samples=800 \
    training.max_steps=100000 \
    training.fp16=True \
    training.output_dir=./exp/checkpoints/$TIMESTAMP \
    training.output_log_dir=./exp/logs/$TIMESTAMP \
    model.cache_dir=/mnt/blob/weights/.model.cache/ \
    training.save_strategy=steps \
    training.save_steps=5000 \
    training.save_total_limit=3 \
    training.optim=adamw_torch \
    training.evaluate_before_train=True \
    training.per_device_train_batch_size=1 \
    training.evaluation_strategy=steps \
    training.eval_steps=5000 \
    training.logging_steps=500 \
    training.logging_first_step=True \
    training.dataloader_num_workers=4 \
    training.num_masks_per_sample=16 \
    wandb.project=train_script \
    wandb.name=$WANDB_NAME \
    model.num_caption_tokens=8 \
    model.additional_num_hidden_layers=12 \
    model.num_task_tokens=6 \
    training.lr_scheduler_type=cosine \
    model.lm_head_model_name_or_path=StanfordAIMI/RadLLaMA-7b \
    training.learning_rate=1e-4 \
    training.weight_decay=1e-4 \
    training.warmup_steps=200 \
    training.warmup_ratio=0.33333333 \
    training.compute_metrics=True \
    +data_transforms=lsj-1_0-1_0 \
    model.sam_model_name_or_path=facebook/sam-vit-base