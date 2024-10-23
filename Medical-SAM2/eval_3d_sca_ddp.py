# eval.py
#!/usr/bin/env python3

"""Evaluate network using PyTorch
   Yunli Qi
"""

import os
import time
import torch
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


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

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)

    if args.weight_path:
        print(f"Loading weights from {args.weight_path}")
        weights = torch.load(args.weight_path, map_location='cpu')['model']
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

        # print(f'weights: {len(weights.keys())}')
        # print(f'net.state_dict(): {len(net.state_dict().keys())}')
        # import time
        # time.sleep(100)

    else:
        raise ValueError("Please specify the weight file to load using --weight_path")

    net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(GPUdevice).major >= 8:
        # Turn on TensorFloat-32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Get test data loader
    _, nice_test_loader = get_dataloader(args, )

    writer = None
    if rank == 0:
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

        # Use TensorBoard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

        writer.add_text('Eval_config', str(args), 0)

    '''Begin evaluation'''
    net.eval()
    global_step = 0  # Or set to appropriate value if needed
    caption_loss = function.validation_sca_ddp(args, nice_test_loader, global_step, global_step, net, writer)

    if rank == 0:
        logger.info(f'Caption loss: {caption_loss} || @ step {global_step}.')

    if rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    args = cfg.parse_args()
    # Ensure that args.load_weight is available
    mp.spawn(main, args=(world_size, args,), nprocs=world_size)
