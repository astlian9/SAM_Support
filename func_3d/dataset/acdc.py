""" Dataloader for the ACDC dataset
    Yang Xing
"""
import os
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox, random_click_new


class ACDC(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):

        # Set the data list for training
        if mode == 'Training':
            self.data_path = os.path.join(data_path, 'train')
            self.mask_path = os.path.join(data_path, 'train_mask')
        if mode == 'Testing':
            self.data_path = os.path.join(data_path, 'test')
            self.mask_path = os.path.join(data_path, 'test_mask')

        # Set the basic information of the dataset
        self.name_list = os.listdir(self.data_path)
        self.name_list = [i[:19] for i in self.name_list]
        self.name_list = list(set(self.name_list))
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.out_size = args.out_size
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        # if self.mode == 'Training':
        #     return 100
        # if self.mode == 'Testing':
        #     return 50
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        # if self.mode == 'Training':
        #     name = 'patient'+str(index).zfill(3)+'_frame01_'
        # if self.mode == 'Testing':
        #     name = 'patient'+str(index+100)+'_frame01_'
        name = self.name_list[index]
        mask_name = '_MRI_heart_right+heart+ventricle.png'
        # mask_name = '_MRI_heart_left+heart+ventricle.png'
        # mask_name = '_MRI_heart_myocardium.png'
        img = np.zeros((10, 1024, 1024, 3))
        mask = np.zeros((10, 1024, 1024))
        for i in range(10):
            img_file_name = name+str(i)+'_MRI_heart.png'
            mask_file_name = name+str(i)+mask_name
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

        point_label, pt = random_click(np.array(mask), point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'mask': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }