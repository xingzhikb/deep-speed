#!/usr/bin/env python3

""" Train network using Hugging Face Trainer with DeepSpeed """

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup
from tensorboardX import SummaryWriter
import wandb

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader

from huggingface_hub import login

# Login to Hugging Face Hub
login(token='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')

# Custom Trainer class
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Step 1: Freeze certain layers
        sam_layers = (
            []
            + list(self.model.image_encoder.parameters())
            + list(self.model.sam_prompt_encoder.parameters())
            + list(self.model.sam_mask_decoder.parameters())
        )
        mem_layers = (
            []
            + list(self.model.obj_ptr_proj.parameters())
            + list(self.model.memory_encoder.parameters())
            + list(self.model.memory_attention.parameters())
            + list(self.model.mask_downsample.parameters())
        )
        for param in sam_layers:
            param.requires_grad = False
        for param in mem_layers:
            param.requires_grad = False

        # Step 2: Identify the layers to train
        llm_layers = (
            []
            + list(self.model.language_model.parameters())
        )
        text_projector_layers = (
            []
            + list(self.model.caption_tokens.parameters())
            + list(self.model.sam_text_transformer.parameters())
            + list(self.model.language_project.parameters())
        )

        # Step 3: Set up the optimizer with the parameters to train
        self.optimizer = optim.Adam([
            {
                'params': text_projector_layers,
                'lr': 1e-4,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 1e-4,
                'amsgrad': False
            },
            {
                'params': llm_layers,
                'lr': 1e-6,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 1e-4,
                'amsgrad': False
            },
        ])

        # Step 4: Set up the learning rate scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss computation logic (same as before)
        imgs_tensor = inputs['image']
        mask_dict = inputs['label']
        if self.args.prompt == 'click':
            pt_dict = inputs['pt']
            point_labels_dict = inputs['p_label']
        elif self.args.prompt == 'bbox':
            bbox_dict = inputs['bbox']

        imgs_tensor = imgs_tensor.squeeze(0)
        imgs_tensor = imgs_tensor.to(dtype=torch.float32, device=model.device)

        train_state = model.train_init_state(
            imgs_tensor=imgs_tensor,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )

        prompt_frame_id = list(range(0, self.args.video_length, self.args.prompt_freq))
        obj_list = []
        for id in prompt_frame_id:
            obj_list += list(mask_dict[id].keys())
        obj_list = list(set(obj_list))
        if len(obj_list) == 0:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            return (loss, None) if return_outputs else loss

        try:
            for id in prompt_frame_id:
                for ann_obj_id in obj_list:
                    if self.args.prompt == 'click':
                        points = pt_dict[id][ann_obj_id].to(device=model.device)
                        labels = point_labels_dict[id][ann_obj_id].to(device=model.device)
                        _, _, _ = model.train_add_new_points(
                            inference_state=train_state,
                            frame_idx=id,
                            obj_id=ann_obj_id,
                            points=points,
                            labels=labels,
                            clear_old_points=False,
                        )
                    elif self.args.prompt == 'bbox':
                        bbox = bbox_dict[id][ann_obj_id]
                        _, _, _ = model.train_add_new_bbox(
                            inference_state=train_state,
                            frame_idx=id,
                            obj_id=ann_obj_id,
                            bbox=bbox.to(device=model.device),
                            clear_old_points=False,
                        )
        except KeyError:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            return (loss, None) if return_outputs else loss

        video_segments = {}
        caption_tokens_segments = {}
        batch_caption_tokens = []

        for out_frame_idx, out_obj_ids, out_mask_logits, caption_tokens in model.train_propagate_in_video(train_state, start_frame_idx=0):
            video_segments[out_frame_idx] = {
                out_obj_id: out_mask_logits[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            caption_tokens_segments[out_frame_idx] = caption_tokens

        for id in range(self.args.video_length):
            caption_tokens = caption_tokens_segments[id]
            num_objs, num_total_caption_tokens, hidden_size = caption_tokens.shape
            batch_caption_tokens.append(caption_tokens.reshape(-1, num_objs * num_total_caption_tokens, hidden_size))

        batch_caption_tokens = torch.cat(batch_caption_tokens, dim=1)
        batch_caption_loss, batch_caption_logits = model.compute_caption_loss_in_video(train_state, batch_caption_tokens)

        # Reset state
        model.reset_state(train_state)

        if return_outputs:
            return batch_caption_loss, batch_caption_logits
        else:
            return batch_caption_loss

def main(args):
    # Initialize WandB
    if args.local_rank == 0:
        wandb_run = wandb.init(
            project="med_sca2",
            config=args,
        )

    # Get the network without moving it to GPU yet
    net, net_cfg = get_network(
        args=args, 
        net=args.net, 
        use_gpu=torch.cuda.is_available(), 
        gpu_device=args.local_rank,  # Local rank corresponds to the GPU device ID
        distribution=True  # Enable distribution to handle multiple GPUs
    )


    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain, map_location='cpu')  # Load weights on CPU
        net.load_state_dict(weights, strict=False)

    if args.resume:
        print(args.resume)
        ckpts = torch.load(args.resume, map_location='cpu')  # Load checkpoints on CPU
        net.load_state_dict(ckpts["model"], strict=False)
        del ckpts

    # Load data (ensure DataLoader uses DistributedSampler for DDP)
    nice_train_loader, nice_test_loader = get_dataloader(args, net_cfg)
    train_dataset = nice_train_loader.dataset
    eval_dataset = nice_test_loader.dataset

    # Calculate total training steps
    total_training_steps = settings.EPOCH * len(nice_train_loader)
    warmup_steps = int(len(nice_train_loader) * 0.01)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=settings.EPOCH,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=warmup_steps,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=args.val_step_freq,
        save_steps=args.val_step_freq,
        save_total_limit=2,
        fp16=True,
        deepspeed=args.deepspeed,  # Path to your DeepSpeed config
        report_to=["wandb"],
        load_best_model_at_end=True,
        dataloader_num_workers=args.num_workers,
        local_rank=args.local_rank,  # Pass local rank to Hugging Face Trainer
    )

    # Initialize Trainer without moving the model to GPU beforehand
    trainer = CustomTrainer(
        model=net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,  # Add tokenizer if needed
    )

    # Start training
    trainer.train(resume_from_checkpoint=args.resume)

    # Finish WandB
    wandb.finish()

if __name__ == '__main__':
    args = cfg.parse_args()
    main(args)
