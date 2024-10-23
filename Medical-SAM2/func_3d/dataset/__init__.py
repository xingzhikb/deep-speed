from .btcv import BTCV
from .amos import AMOS
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .m3d_seg import M3DSegDataset
from .m3d_cap import M3DCapDataset
from transformers import AutoTokenizer

from torch.utils.data.distributed import DistributedSampler


def get_dataloader(args, net_cfg):
   
    
    if args.dataset == 'm3d_seg':
        '''m3d data'''
        tokenizer = AutoTokenizer.from_pretrained(net_cfg['model']['language_model_name_or_path'])

        if tokenizer.eos_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': tokenizer.pad_token})
        # special_token = {"additional_special_tokens": ["<prompt_token>"]}
        # tokenizer.add_special_tokens(
        #     special_token
        # )
        
        m3d_train_dataset = M3DSegDataset(args, args.data_path, tokenizer, args.max_length, transform = None, transform_msk= None, tag = args.tag, mode = 'train', prompt=args.prompt)
        m3d_test_dataset = M3DSegDataset(args, args.data_path, tokenizer, args.max_length, transform = None, transform_msk= None, tag = args.tag, mode = 'test', prompt=args.prompt)

        # nice_train_loader = DataLoader(m3d_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        # nice_test_loader = DataLoader(m3d_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'm3d_cap':
        '''m3d data'''
        tokenizer = AutoTokenizer.from_pretrained(net_cfg['model']['language_model_name_or_path'])
        if tokenizer.eos_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # special_token = {"additional_special_tokens": ["<prompt_token>"]}
        # tokenizer.add_special_tokens(
        #     special_token
        # )
        
        m3d_train_dataset = M3DCapDataset(args, args.data_path, tokenizer, args.max_length, transform = None, transform_msk= None, mode = 'train', prompt=args.prompt, variation=0, seed=42)
        m3d_test_dataset = M3DCapDataset(args, args.data_path, tokenizer, args.max_length, transform = None, transform_msk= None, mode = 'test', prompt=args.prompt)

        # nice_train_loader = DataLoader(m3d_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        # nice_test_loader = DataLoader(m3d_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    else:
        
        raise NotImplementedError("the dataset is not supported now!!!")

    if 'm3d' in args.dataset:
        if args.use_ddp:
            nice_train_loader = DataLoader(m3d_train_dataset, batch_size=1, shuffle=False, pin_memory=True, sampler=DistributedSampler(m3d_train_dataset))
            nice_test_loader = DataLoader(m3d_test_dataset, batch_size=1, shuffle=False, pin_memory=True, sampler=DistributedSampler(m3d_test_dataset))
        else:
            nice_train_loader = DataLoader(m3d_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
            nice_test_loader = DataLoader(m3d_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        
    return nice_train_loader, nice_test_loader