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

from func_3d.utils import random_click, generate_bbox
from monai.data import load_decathlon_datalist

from .prompt_templates import Seg_templates
from .dataset_info import dataset_info
from .term_dictionary import term_dict


class M3DSegDataset(Dataset):
    def __init__(self, args, data_path, tokenizer = None, max_length = None, transform = None, transform_msk = None, tag = '0000', mode = "train", prompt = 'click', description = False, seed=None, variation=0):

        # Set the data list for training
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=data_path,
                data_list_file_path=os.path.join(data_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=data_path,
                data_list_file_path=os.path.join(data_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=data_path,
                data_list_file_path=os.path.join(data_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        
        self.multimask_output = args.multimask_output
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_caption_tokens = args.num_caption_tokens
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.tag = tag
        self.dataset_info = dataset_info
        self.description = description
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
        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        point_label = 1
        # newsize = (self.img_size, self.img_size)

        """Get the images"""
        data = self.data_list[index]
        image_path = data["image"]
        mask_path = data["label"]
        
        image_array = np.load(image_path)  # 1*512*512*L
        masks_array= np.load(mask_path)
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
        cls_list = self.dataset_info[self.tag]
        # vld_cls = torch.nonzero(torch.sum(masks_array, dim=(1, 2, 3))).flatten().tolist()
        vld_cls = True
        # mask_ids_list = list(np.unique(masks_array[masks_array > 0]))
        cls_id = 1

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
                # obj_mask = obj_mask.resize(newsize)
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
            # image = image.resize(newsize)
            image = torch.tensor(np.array(image)).permute(2, 0, 1)

            image_tensor[frame_index - starting_frame, :, :, :] = image
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        if vld_cls:
            if not self.description:
                question_temple = random.choice(self.cls_questions)
                question = question_temple.format(cls_list[cls_id])
                question = "<prompt_token>" * self.multimask_output * self.num_caption_tokens + question
                answer = random.choice(self.cls_answers)
            else:
                question_temple = random.choice(self.des_questions)
                question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                question = "<prompt_token>" * self.multimask_output * self.num_caption_tokens + question
                answer = random.choice(self.des_answers).format(cls_list[cls_id])
        else:
            if not self.description:
                question_temple = random.choice(self.cls_questions)
                question = question_temple.format(cls_list[cls_id])
                question = "<prompt_token>" * self.multimask_output * self.num_caption_tokens + question
                answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
            else:
                question_temple = random.choice(self.des_questions)
                question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                question = "<prompt_token>" * self.multimask_output * self.num_caption_tokens + question
                answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

        text_tensor = self.tokenizer(
            answer,
            truncation=True,
            max_length = self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        labels = input_ids.clone()
        labels[valid_len+1:] = -100 # add an eos token for eos_token == pad_token

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
            }