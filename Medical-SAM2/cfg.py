import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='samba_train_test', type=str, help='experiment name')
    parser.add_argument('-vis', type=bool, default=False, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='bbox', help='type of prompt, bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
    parser.add_argument('-bbox_variation', type=int, default=100, help='bbox variations of x and y')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-resume', type=str, default=None, help='path of resume weights')
    parser.add_argument('-val_freq',type=int,default=1,help='interval between each validation')
    parser.add_argument('-val_step_freq',type=int,default=5000,help='interval global step between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-use_ddp', default=False ,type=bool,help='whether use DDP or not')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='btcv' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=2, help='video length')
    parser.add_argument('-max_length', type=int, default=64, help='tokenization max length')
    parser.add_argument('-cap_data_path', type=str, default=None , help='cap data json path')
    parser.add_argument('-num_caption_tokens', type=int, default=8, help='caption tokens number for each mask')
    parser.add_argument('-tag', type=str, default="0000", help='m3d dataset tag')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
    parser.add_argument(
    '-data_path',
    type=str,
    default='./data/btcv',
    help='The path of segmentation data')
    parser.add_argument(
    '-weight_path',
    type=str,
    default=None,
    help='The path of segmentation model weights')
    parser.add_argument('-deepspeed', type=str, default=None, help='Path to DeepSpeed config')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of workers for DataLoader')
    opt = parser.parse_args()
    return opt
