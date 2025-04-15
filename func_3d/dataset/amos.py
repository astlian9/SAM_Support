""" Dataloader for the amos22 dataset
    Yang Xing
"""
import os
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox, random_click_new


class AMOS(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        if mode == 'Training':
            self.data_path = os.path.join(data_path, 'train')
            self.mask_path = os.path.join(data_path, 'train_mask')
            self.index_path = os.path.join(data_path,'train_index.csv')
        if mode == 'Testing':
            self.data_path = os.path.join(data_path, 'test')
            self.mask_path = os.path.join(data_path, 'test_mask')
            self.index_path = os.path.join(data_path, 'test_index.csv')

        # self.organ = 'duodenum'
        # self.organ = 'postcava'
        self.organ = args.task
        print(self.organ)
        # Set the basic information of the dataset
        index = pd.read_csv(self.index_path,dtype=str)
        index = index[(index['organ']==self.organ)].copy()
        index['mask_name'] = index['dataset']+'_'+index['id']+'_'+index['slice']+'_'+index['type']+'_'+index['task']+'_'+index['organ']+'.png'
        index['image_name'] = index['dataset'] + '_' + index['id'] + '_' + index['slice'] + '_' + index['type'] + '_' + index['task'] + '.png'
        self.index = index
        self.name_list = index['id'].unique()
        # if mode=='Training':
        #     self.name_list=['0001','0004']
        # if mode=='Testing':
        #     self.name_list=['0304','0323','0344','0308','0325']
        self.mode = mode

        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.out_size = args.out_size
        self.video_length = args.video_length

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        newsize = (self.img_size, self.img_size)

        """Get the images"""

        name = self.name_list[index]
        if self.mode=='Training':
            video_length = self.video_length
            if len(self.index[self.index['id']==name]) < self.video_length:
                start_frame=0
            else:
                start_frame = np.random.randint(0,len(self.index[self.index['id']==name]) - video_length+1)
        if self.mode == 'Testing':
            video_length = min(len(self.index[self.index['id']==name]),55)
            # video_length = 30
            start_frame = 0
        img_file_list = self.index[self.index['id']==name]['image_name'].to_list()
        mask_file_list = self.index[self.index['id']==name]['mask_name'].to_list()
        img = np.zeros((video_length, 1024, 1024, 3))
        mask = np.zeros((video_length, 1024, 1024))
        name_list = []
        for i in range(min(video_length, len(img_file_list))):
            img_file_name = img_file_list[i+start_frame]
            name_list.append(img_file_name)
            mask_file_name = mask_file_list[i+start_frame]
            if os.path.exists(os.path.join(self.data_path,img_file_name)):
                image = Image.open(os.path.join(self.data_path,img_file_name)).convert('RGB')
                img[i] = np.array(image) / 255
                if os.path.exists(os.path.join(self.mask_path,mask_file_name)):
                    mask_image = Image.open(os.path.join(self.mask_path,mask_file_name)).convert('L')
                    mask[i] = np.array(mask_image) / 255

        # img_tensor = torch.zeros(9, 3, self.img_size, self.img_size)
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(1)
        img = F.interpolate(img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

        point_label, pt = random_click(np.array(mask), 1)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

        image_meta_dict = {'filename_or_obj': name_list}
        return {
            'image': img,
            'mask': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }