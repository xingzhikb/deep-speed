""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from func_3d.utils import eval_seg

import wandb
import evaluate

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
args = cfg.parse_args()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler('cuda')
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch, writer):
    hard = 0
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for step, pack in enumerate(train_loader):
            global_step = len(train_loader) * epoch + step
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5
            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                writer.add_scalar('Training_loss', loss, global_step)
                writer.add_scalar('Training_prompt_loss', prompt_loss, global_step)
                writer.add_scalar('Training_non_prompt_loss', non_prompt_loss, global_step)

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step()
                
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.vis:
                            os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy())
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/val/{name[0]}/{id}/{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        loss += lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])


def train_sca(args, net: nn.Module, optimizer1, optimizer2, optimizer3, train_loader, test_loader,
          epoch, writer):
    hard = 0
    epoch_loss = 0
    epoch_caption_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    if optimizer3 is not None:
        optimizer3.zero_grad()
    video_length = args.video_length

    if getattr(args, "rank", None) == None:
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        args.rank = args.gpu_device
    else:
        GPUdevice = torch.device('cuda:' + str(args.rank))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', disable=(args.rank!=0)) as pbar:
        for step, pack in enumerate(train_loader):
            global_step = len(train_loader) * epoch + step
        #iterate each 3d volume
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.module.train_init_state(imgs_tensor=imgs_tensor, input_ids=pack["input_ids"], attention_mask=pack["attention_mask"], labels=pack["labels"])
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5
            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.module.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.module.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.module.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                batch_caption_tokens = []

                for out_frame_idx, out_obj_ids, out_mask_logits, caption_tokens in net.module.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    # for i, out_obj_id in enumerate(out_obj_ids):
                    #     if out_obj_id in batch_caption_tokens:
                    #         batch_caption_tokens[out_obj_id] += [caption_tokens[i].unsqueeze(0)] # [B, P, num_output_heads, self.num_caption_tokens, D] NOTE: P == num_masks
                    #     else:
                    #         batch_caption_tokens[out_obj_id] = []
                    batch_size, num_masks, num_output_heads, num_caption_tokens, hidden_size = caption_tokens.shape
                    caption_tokens = caption_tokens.reshape(
                        -1, batch_size * num_masks * num_output_heads * num_caption_tokens, hidden_size
                    )
                    batch_caption_tokens.append(caption_tokens)


                # batch_caption_tokens = [torch.cat(v, dim=0).mean(dim=0) for _, v in batch_caption_tokens.items()] #[1, num_output_heads, num_caption_tokens, D]
                # batch_caption_tokens = torch.cat(batch_caption_tokens, dim=0) # [1, num_output_heads, num_caption_tokens, D]
                batch_caption_tokens = torch.cat(batch_caption_tokens, dim=1) # [1, sum(num_masks* num_output_heads * num_caption_tokens), num_caption_tokens, D]
                batch_caption_loss, batch_caption_logits = net.module.compute_caption_loss_in_video(train_state, batch_caption_tokens)
                epoch_caption_loss += batch_caption_loss.item()

                # tensorboard writer add loss
                if args.rank == 0: writer.add_scalar('Training_loss', batch_caption_loss.item(), global_step)

                
                if optimizer3 is not None:
                    batch_caption_loss.backward()
                    optimizer3.step()

                # loss = 0
                # non_prompt_loss = 0
                # prompt_loss = 0
                # for id in range(video_length):
                #     for ann_obj_id in obj_list:
                #         pred = video_segments[id][ann_obj_id]
                #         pred = pred.unsqueeze(0)
                #         # pred = torch.sigmoid(pred)
                #         try:
                #             mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                #         except KeyError:
                #             mask = torch.zeros_like(pred).to(device=GPUdevice)
                #         if args.train_vis:
                #             os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                #             fig, ax = plt.subplots(1, 3)
                #             ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
                #             ax[0].axis('off')
                #             ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                #             ax[1].axis('off')
                #             try:
                #                 bbox = bbox_dict[id][ann_obj_id]
                #                 ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                #             except KeyError:
                #                 pass
                #             ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                #             ax[2].axis('off')
                #             plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                #             plt.close()
                #         obj_loss = lossfunc(pred, mask)
                #         loss += obj_loss.item()
                #         if id in prompt_frame_id:
                #             prompt_loss += obj_loss
                #         else:
                #             non_prompt_loss += obj_loss
                # loss = loss / video_length / len(obj_list)
                # if prompt_freq > 1:
                #     non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                # prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                # pbar.set_postfix(**{'loss (batch)': loss})
                # epoch_loss += loss
                # epoch_prompt_loss += prompt_loss.item()
                # if prompt_freq > 1:
                #     epoch_non_prompt_loss += non_prompt_loss.item()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                #     non_prompt_loss.backward(retain_graph=True)
                #     optimizer2.step()
                # if optimizer1 is not None:
                #     prompt_loss.backward()
                #     optimizer1.step()
                
                #     optimizer1.zero_grad()
                # if optimizer2 is not None:
                #     optimizer2.zero_grad()

                if optimizer3 is not None:
                    optimizer3.zero_grad()

                net.module.reset_state(train_state)
            
            pbar.update()
            if global_step % args.val_step_freq == 0:
                caption_loss = validation_sca(args, test_loader, epoch, global_step, net, writer)
                print(f'Caption loss: {caption_loss} || @ epoch {epoch} step {global_step}.')
                net.train()
    # return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader), epoch_caption_loss / len(train_loader)
    return epoch_caption_loss / len(train_loader)


def train_sca_ddp(args, net: nn.Module, optimizer, lr_scheduler, train_loader, test_loader,
          epoch, writer):
    hard = 0
    epoch_loss = 0
    epoch_caption_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    # if optimizer1 is not None:
    #     optimizer1.zero_grad()
    # if optimizer2 is not None:
    #     optimizer2.zero_grad()
    if optimizer is not None:
        optimizer.zero_grad()
    video_length = args.video_length

    if getattr(args, "rank", None) == None:
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        args.rank = args.gpu_device
    else:
        GPUdevice = torch.device('cuda:' + str(args.rank))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', disable=(args.rank!=0)) as pbar:
        for step, pack in enumerate(train_loader):
            global_step = len(train_loader) * epoch + step
            if args.resume:
                pbar.update()
                if step <= args.resume_step:
                    continue
            #iterate each 3d volume
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.module.train_init_state(imgs_tensor=imgs_tensor, input_ids=pack["input_ids"], attention_mask=pack["attention_mask"], labels=pack["labels"])
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5
            with torch.cuda.amp.autocast():
                try:
                    for id in prompt_frame_id:
                        for ann_obj_id in obj_list:
                            # try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.module.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.module.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                except KeyError:
                    continue    
                        # except KeyError:
                        #     _, _, _ = net.module.train_add_new_mask(
                        #         inference_state=train_state,
                        #         frame_idx=id,
                        #         obj_id=ann_obj_id,
                        #         mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                        #     )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                caption_tokens_segments = {}
                batch_caption_tokens = []

                for out_frame_idx, out_obj_ids, out_mask_logits, caption_tokens in net.module.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    # batch_size, num_masks, num_output_heads, num_caption_tokens, hidden_size = caption_tokens.shape
                    # caption_tokens = caption_tokens.reshape(
                    #     -1, batch_size * num_masks * num_output_heads * num_caption_tokens, hidden_size
                    # )
                    # batch_caption_tokens.append(caption_tokens)
                    caption_tokens_segments[out_frame_idx] = caption_tokens

                for id in range(video_length):
                    caption_tokens= caption_tokens_segments[id]
                    num_objs, num_total_caption_tokens, hidden_size = caption_tokens.shape
                    batch_caption_tokens.append(caption_tokens.reshape(-1, num_objs * num_total_caption_tokens, hidden_size))

                # batch_caption_tokens = [torch.cat(v, dim=0).mean(dim=0) for _, v in batch_caption_tokens.items()] #[1, num_output_heads, num_caption_tokens, D]
                # batch_caption_tokens = torch.cat(batch_caption_tokens, dim=0) # [1, num_output_heads, num_caption_tokens, D]
                batch_caption_tokens = torch.cat(batch_caption_tokens, dim=1) # [1, sum(num_objs * num_total_caption_tokens), D]
                batch_caption_loss, batch_caption_logits = net.module.compute_caption_loss_in_video(train_state, batch_caption_tokens)
                epoch_caption_loss += batch_caption_loss.item()

                # tensorboard writer add loss
                if args.rank == 0: 
                    writer.add_scalar('Training_loss', batch_caption_loss.item(), global_step)
                    wandb.log({"Training_loss": batch_caption_loss.item()})
                
                if optimizer is not None:
                    batch_caption_loss.backward()
                    optimizer.step()

                lr_scheduler.step()

                net.module.reset_state(train_state)
            
            pbar.update()
            if global_step % args.val_step_freq == 0 and global_step != 0:
                if args.rank == 0: 
                    torch.save(
                        {'model': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()}, 
                         os.path.join(args.path_helper['ckpt_path'], f'step-{global_step}.pth'))
                validation_sca_ddp(args, test_loader, epoch, global_step, net, writer)
                net.train()

    return epoch_caption_loss / len(train_loader)

def validation_sca_ddp(args, val_loader, epoch, step, net: nn.Module, writer):
    net.eval()

    n_val = len(val_loader)
    prompt_freq = args.prompt_freq
    prompt = args.prompt
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    all_decoded_preds = []
    all_decoded_labels = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, disable=(rank != 0)) as pbar:
        for i, pack in enumerate(val_loader):
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))

            train_state = net.module.val_init_state(
                imgs_tensor=imgs_tensor,
                input_ids=pack["input_ids"],
                attention_mask=pack["attention_mask"],
                labels=pack["labels"]
            )
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                net.module.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                net.module.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            net.module.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                batch_caption_tokens = []

                for out_frame_idx, out_obj_ids, out_mask_logits, caption_tokens in net.module.propagate_in_video(
                    train_state, start_frame_idx=0):
                    num_objs, num_total_caption_tokens, hidden_size = caption_tokens.shape
                    caption_tokens = caption_tokens.reshape(
                        -1, num_objs * num_total_caption_tokens, hidden_size
                    )
                    batch_caption_tokens.append(caption_tokens)

                if batch_caption_tokens:
                    batch_caption_tokens = torch.cat(batch_caption_tokens, dim=1)
                    decoded_preds, decoded_labels = net.module.compute_caption_loss_in_video_val(
                        inference_state = train_state, batch_caption_tokens = batch_caption_tokens, tokenizer = val_loader.dataset.tokenizer, answer = pack['answer'])
                    # print(decoded_preds, decoded_labels)
                    
                    # Collect predictions and labelstokenizer
                    all_decoded_preds.extend(decoded_preds)
                    all_decoded_labels.extend(decoded_labels)

            net.module.reset_state(train_state)
            pbar.update()

    # Now gather predictions and labels from all processes
    gathered_decoded_preds = [None for _ in range(world_size)]
    gathered_decoded_labels = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_decoded_preds, all_decoded_preds)
    dist.all_gather_object(gathered_decoded_labels, all_decoded_labels)

    if rank == 0:
        # Concatenate the lists
        total_decoded_preds = []
        total_decoded_labels = []
        for preds in gathered_decoded_preds:
            if preds:
                total_decoded_preds.extend(preds)
        for labels in gathered_decoded_labels:
            if labels:
                total_decoded_labels.extend(labels)
        # Compute metrics
        eval_result = compute_metrics(total_decoded_preds, total_decoded_labels)
        # Log the metrics
        if step is not None:
            wandb.log(eval_result)
    else:
        eval_result = None


def validation_sca(args, val_loader, epoch, step, net: nn.Module, writer):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt
    epoch_caption_loss = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.module.val_init_state(imgs_tensor=imgs_tensor, input_ids=pack["input_ids"], attention_mask=pack["attention_mask"], labels=pack["labels"])
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.module.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.module.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.module.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                batch_caption_tokens = []

                for out_frame_idx, out_obj_ids, out_mask_logits, caption_tokens in net.module.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    batch_size, num_masks, num_output_heads, num_caption_tokens, hidden_size = caption_tokens.shape
                    caption_tokens = caption_tokens.reshape(
                        -1, batch_size * num_masks * num_output_heads * num_caption_tokens, hidden_size
                    )
                    batch_caption_tokens.append(caption_tokens)

                # batch_caption_tokens = [torch.cat(v, dim=0).mean(dim=0) for _, v in batch_caption_tokens.items()] #[1, num_output_heads, num_caption_tokens, D]
                batch_caption_tokens = torch.cat(batch_caption_tokens, dim=1) # [1, num_output_heads, num_caption_tokens, D]
                decoded_preds, decoded_labels = net.module.compute_caption_loss_in_video_val(train_state, batch_caption_tokens)
                eval_result = compute_metrics(decoded_preds, decoded_labels)
                wandb.log(eval_result)

            net.module.reset_state(train_state)
            pbar.update()
    if step != None:
        # tensorboard writer add loss
        writer.add_scalar('Validation_loss', epoch_caption_loss / len(val_loader), step)
    # return tot/ n_val , tuple([a/n_val for a in mix_res]), epoch_caption_loss / len(val_loader)
    return epoch_caption_loss / len(val_loader)
def compute_metrics(decoded_preds, decoded_labels):
    
    result = dict()
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
    result["bleu"] = bleu_score['bleu']

    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
    result["rouge1"] = rouge_score['rouge1']

    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    result["meteor"] = meteor_score['meteor']

    # bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    # result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])
    return result