""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import random
import numpy as np
from scipy import sparse
import ast
import torch
from PIL import Image
from torch.utils.data import Dataset
import json

from func_3d.utils import random_click, generate_bbox
from monai.data import load_decathlon_datalist

from .prompt_templates import Seg_templates, Caption_templates
from .dataset_info import dataset_info
from .term_dictionary import term_dict


class M3DCapDataset(Dataset):
    def __init__(self, args, data_path, tokenizer = None, max_length = None, transform = None, transform_msk = None, mode = "train", prompt = 'click', seed=None, variation=0):
        
        
        self.multimask_output = args.multimask_output
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_caption_tokens = args.num_caption_tokens
        # Set the basic information of the dataset
        self.args = args
        self.data_path = data_path
        self.mode = mode
        self.caption_prompts = Caption_templates
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'train':
            self.video_length = args.video_length
        else:
            self.video_length = None

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)
        max_attempts = 100
        for _ in range(max_attempts):
            """Get the images"""
            try:
                data = self.data_list[index]
                image_path = data["image"]

                image_abs_path = os.path.join(self.data_path, image_path)
                
                image_array = np.load(image_abs_path)  # 1*L*512*512
                image_array = np.swapaxes(image_array, -1, -3) # 1*512*512*L
                masks_array = np.where(image_array > 0, 1, 0).astype(bool)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_path, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text
                # masks_array= np.load(mask_path)
                # num_frame = masks_array.shape[-1]
                # masks_array = np.zeros(data_seg_3d_shape + (num_frame,))
                # for i in range(num_frame):
                #     masks_array[..., i] = np.load(os.path.join(mask_path, f'{i}.npy'))
                for i in range(masks_array.shape[-1]):
                    if np.sum(masks_array[..., i]) > 0:
                        masks_array = masks_array[..., i:]
                        break
                starting_frame_nonzero = i
                for j in reversed(range(masks_array.shape[-1])):
                    if np.sum(masks_array[..., j]) > 0:
                        masks_array = masks_array[..., :j+1]
                        break
                num_frame = masks_array.shape[-1]
                if self.video_length is None:
                    video_length = int(num_frame / 4)
                else:
                    video_length = self.video_length
                if num_frame > video_length and self.mode == 'train':
                    starting_frame = np.random.randint(0, num_frame - video_length + 1)
                else:
                    starting_frame = 0
                image_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
                mask_dict = {}
                point_label_dict = {}
                pt_dict = {}
                bbox_dict = {}

                for frame_index in range(starting_frame, starting_frame + video_length):
                    # image = Image.open(os.path.join(image_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
                    image = Image.fromarray(image_array[0, :, :, frame_index + starting_frame_nonzero]).convert('RGB')
                    mask = masks_array[0, ..., frame_index]
                    # mask = np.rot90(mask)
                    obj_list = np.unique(mask[mask > 0])
                    diff_obj_mask_dict = {}
                    if self.prompt == 'bbox':
                        diff_obj_bbox_dict = {}
                    elif self.prompt == 'click':
                        diff_obj_pt_dict = {}
                        diff_obj_point_label_dict = {}
                    else:
                        raise ValueError('Prompt not recognized')
                    for obj in obj_list:
                        obj_mask = mask == obj
                        # if self.transform_msk:
                        obj_mask = Image.fromarray(obj_mask)
                        obj_mask = obj_mask.resize(newsize)
                        obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                            # obj_mask = self.transform_msk(obj_mask).int()
                        diff_obj_mask_dict[obj] = obj_mask

                        if self.prompt == 'click':
                            diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                        if self.prompt == 'bbox':
                            diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
                    # if self.transform:
                        # state = torch.get_rng_state()
                        # image = self.transform(image)
                        # torch.set_rng_state(state)
                    image = image.resize(newsize)
                    image = torch.tensor(np.array(image)).permute(2, 0, 1)

                    image_tensor[frame_index - starting_frame, :, :, :] = image
                    mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
                    if self.prompt == 'bbox':
                        bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
                    elif self.prompt == 'click':
                        pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                        point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

                text_tensor = self.tokenizer(
                    answer,
                    padding=True,
                    truncation=True,
                    # max_length = self.max_length,
                    # padding="max_length",
                    return_tensors="pt"
                )

                input_ids = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                # valid_len = torch.sum(attention_mask)
                labels = input_ids.clone()
                # labels[valid_len+1:] = -100 # add an eos token for eos_token == pad_token

                # answer_tensor = self.tokenizer(
                #     answer,
                #     return_tensors="pt"
                # )

                # valid_len = torch.sum(attention_mask)
                # if valid_len < len(input_ids):
                #     input_ids[valid_len] = self.tokenizer.eos_token_id

                # question_tensor = self.tokenizer(
                #     question, 
                #     max_length=self.tokenizer.model_max_length, 
                #     truncation=True, 
                #     padding="max_length", 
                #     return_tensors="pt"
                # )
                # question_len = torch.sum(question_tensor["attention_mask"][0])

                # labels = input_ids.clone()
                # labels[:question_len] = -100
                # if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                #     labels[labels == self.tokenizer.pad_token_id] = -100
                #     if valid_len < len(labels):
                #         labels[valid_len] = self.tokenizer.eos_token_id
                # else:
                #     labels[labels == self.tokenizer.eos_token_id] = -100

                image_meta_dict = {'filename_or_obj':data}
                if self.prompt == 'bbox':
                    return {
                        'image':image_tensor,
                        'label': mask_dict,
                        'bbox': bbox_dict,
                        'image_meta_dict':image_meta_dict,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels,
                        'answer': answer,
                    }
                elif self.prompt == 'click':
                    return {
                        'image':image_tensor,
                        'label': mask_dict,
                        'p_label':point_label_dict,
                        'pt':pt_dict,
                        'image_meta_dict':image_meta_dict,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels,
                        'answer': answer,
                    }
                
            except Exception as e:
                print(f"Error in __getitem__ at index {index}: {e}")
                index = random.randint(0, len(self.data_list) - 1)