# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers

import wandb
wandb_run = None

from huggingface_hub import login
login(token = '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')
def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):
    ddp_setup(rank, world_size)

    args.rank = rank
    args.use_ddp = True

    GPUdevice = torch.device('cuda', rank)

    net, net_cfg = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.to(dtype=torch.bfloat16)
    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain, map_location=GPUdevice)
        net.load_state_dict(weights,strict=False)

    ckpts = None
    if args.resume and rank==0:
        print(args.resume)
        ckpts = torch.load(args.resume, map_location=GPUdevice)
        net.to("cpu")
        net.load_state_dict(ckpts["model"], strict=False)
        net = net.to(GPUdevice)

    net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    sam_layers = (
                  []
                  + list(net.module.image_encoder.parameters())
                  + list(net.module.sam_prompt_encoder.parameters())
                  + list(net.module.sam_mask_decoder.parameters())
                  )
    mem_layers = (
                  []
                  + list(net.module.obj_ptr_proj.parameters())
                  + list(net.module.memory_encoder.parameters())
                  + list(net.module.memory_attention.parameters())
                  + list(net.module.mask_downsample.parameters())
                  )

    llm_layers = (
        []
        + list(net.module.language_model.parameters())
    )
    text_projector_layers = (
        []
        + list(net.module.caption_tokens.parameters())
        + list(net.module.sam_text_transformer.parameters())
        + list(net.module.language_project.parameters())
    )
    # if len(sam_layers) == 0:
    #     optimizer1 = None
        
    # else:
    #     optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # if len(mem_layers) == 0:
    #     optimizer2 = None
    # else:
    #     optimizer2 = optim.Adam(mem_layers, lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay
    # if len(text_layers) == 0:
    #     optimizer3 =None
    # else:
    for param in sam_layers:
        param.requires_grad = False
    for param in mem_layers:
        param.requires_grad = False
    # for param in llm_layers:
    #     param.requires_grad = False
    optimizer = optim.Adam([
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

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
 
    nice_train_loader, nice_test_loader = get_dataloader(args, net_cfg)

    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(len(nice_train_loader) * 0.01), num_training_steps=int(settings.EPOCH * len(nice_train_loader)))
    if args.resume and ckpts:
        if optimizer:
            optimizer.load_state_dict(ckpts["optimizer"])
        lr_scheduler.load_state_dict(ckpts["lr_scheduler"])  
        del ckpts

    start_epoch = 0
    if args.resume:
        args.resume_step = int(args.resume.split("step-")[1].split(".")[0])
        start_epoch = args.resume_step // len(nice_train_loader)

    writer = None
    if rank == 0:
        global wandb_run
        wandb_run = wandb.init(
            # Set the project where this run will be logged
            project=f"med_sca2",
            # Track hyperparameters and run metadata
            config=args,
        )
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

        #use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.net, settings.TIME_NOW))

        writer.add_text('Training_config', str(args), 0)

        '''checkpoint path and tensorboard'''
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(start_epoch, settings.EPOCH):
        net.train()
        time_start = time.time()
        # loss, prompt_loss, non_prompt_loss, caption_loss = function.train_sca(args, net, optimizer1, optimizer2, optimizer3, nice_train_loader, epoch)
        nice_train_loader.sampler.set_epoch(epoch)
        caption_loss = function.train_sca_ddp(args, net, optimizer, lr_scheduler, nice_train_loader, nice_test_loader, epoch, writer)
        time_end = time.time()
        
        # if rank == 0: 
        #     logger.info(f'Train loss: {caption_loss} || @ epoch {epoch}.')
        #     print('time_for_training ', time_end - time_start)

        net.eval()
        global_step = len(nice_train_loader) * (epoch + 1)
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            # tol, (eiou, edice), caption_loss = function.validation_sca(args, nice_test_loader, epoch, net, writer)
            function.validation_sca_ddp(args, nice_test_loader, epoch, global_step, net, writer)
            
            if rank == 0:
                # logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, caption loss: {caption_loss} || @ epoch {epoch}.')
                # logger.info(f'Caption loss: {caption_loss} || @ epoch {epoch} step {global_step}.')

                torch.save(
                    {'model': net.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict()}, 
                    os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

    if rank == 0: 
        writer.close()
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    # main()
    args = cfg.parse_args()
    mp.spawn(main, args=(world_size, args, ), nprocs=world_size)