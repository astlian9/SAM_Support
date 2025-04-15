""" Dataloader for the PET dataset
    Yang Xing
"""
import os
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox, random_click_new


class PETCT(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        if mode == 'Training':
            self.data_path = os.path.join(data_path, 'train_3d')
        if mode == 'Testing':
            self.data_path = os.path.join(data_path, 'test_label_3d')
            self.mask_path = os.path.join(data_path, 'test_case04_petct_3d')
        
        # Set the basic information of the dataset
        self.name_list = os.listdir(self.data_path)
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
        return len(self.name_list)
        # if self.mode == 'Training':
        #     return 100
        # if self.mode == 'Testing':
        #     return 10
    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        if self.mode=='Training':
            img_path = os.path.join(self.data_path, name)
#         mask_path = os.path.join(self.data_path, name)
            raw_data = np.load(img_path)['arr_0'] #(3,144,144,144)
            data_seg_3d_shape = raw_data.shape
            num_frame = data_seg_3d_shape[-1]
            data_seg_3d_shape = raw_data[1:3]
            data_seg_3d = raw_data[2,:,:,:] #(144,144,144)
        
        if self.mode=='Testing':
            mask_img_path = os.path.join(self.data_path, name)
            img_path = os.path.join(self.mask_path, name[:-4]+'_petct_3d'+name[-4:])
            raw_data = np.load(img_path)['arr_0'][0] #(2,144,144,144)
            raw_mask = np.load(mask_img_path)['arr_0']
            data_seg_3d = raw_mask #(144,144,144)
        
        
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:
                # data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                # data_seg_3d = data_seg_3d[..., :j+1]
                break
        ending_frame_nonzero = j
        # num_frame = data_seg_3d.shape[-1]
        num_frame = ending_frame_nonzero - starting_frame_nonzero
        
        if self.video_length is None:
            video_length = int(num_frame)
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1) + starting_frame_nonzero
        else:
            starting_frame = starting_frame_nonzero
        # img_tensor = torch.zeros(video_length, 2, self.img_size, self.img_size)
        img = torch.from_numpy(raw_data[0, :, :, starting_frame:starting_frame + video_length]).permute(2,0,1)
        mask = torch.from_numpy(data_seg_3d[:, :, starting_frame:starting_frame + video_length]).permute(2,0,1)
        img = img.unsqueeze(1).repeat(1,3,1,1)
        # img = transforms.functional.adjust_gamma(img*255, gamma=0.5) /255
        mask = mask.unsqueeze(1)
        img = F.interpolate(img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

        point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

        image_meta_dict = {'filename_or_obj': name+str(starting_frame_nonzero)}
        return {
            'image': img,
            'mask': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }