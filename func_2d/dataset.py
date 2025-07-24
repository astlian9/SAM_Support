""" train and test dataset

author jundewu
"""
import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import gzip
from PIL import Image
from torch.utils.data import Dataset
from scipy import ndimage

from func_2d.utils import random_click


class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '_cropped.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]

        # img_path = os.path.join(subfolder, name + '.jpg')
        # multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and rater images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        # apply transform
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        # find init click and apply majority vote
        if self.prompt == 'click':

            point_label_cup, pt_cup = random_click(np.array((multi_rater_cup.mean(axis=0)).squeeze(0)), point_label = 1)
            
            selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
            selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 


            selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


            # # Or use any specific rater as GT
            # point_label_cup, pt_cup = random_click(np.array(multi_rater_cup[0, :, :, :].squeeze(0)), point_label = 1)
            # selected_rater_mask_cup_ori = multi_rater_cup[0,:,:,:]
            # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 

            # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }


class STARE(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, prompt='click',
                 plane=False):

        self.data_path = data_path
        self.name_list = os.listdir(os.path.join(data_path, 'masks'))
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index].split('.')[0]

        img_path = os.path.join(self.data_path, 'images', name + '.ppm')

        msk_path = os.path.join(self.data_path, 'masks', name + '.ah.ppm')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask).int()

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'mask': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }


class Pendal(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', plane=False):

        self.args = args
        self.data_path = data_path
        self.name_list = os.listdir(os.path.join(self.data_path,'Images'))
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img = Image.open(os.path.join(self.data_path, 'Images',name)).convert('RGB')
        mask = Image.open(os.path.join(self.data_path, 'Segmentation2',name)).convert('L')

        mask = np.array(mask)
        mask[mask==mask.min()]=0
        mask[mask>0] = 255

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'mask': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


class WBC(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click',
                 plane=False):

        self.data_path = os.path.join(data_path, mode+'_images')
        self.mask_path = os.path.join(data_path, mode+'_masks')
        self.name_list = glob.glob(self.data_path + "/*.bmp")
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1  # available: 1 2

        """Get the images"""
        name = os.path.basename(self.name_list[index])

        img_path = self.name_list[index]
        msk_path = os.path.join(self.mask_path, name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        mask = np.array(mask) // 100
        mask[mask != point_label] = 0
        mask[mask == point_label] = 255

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'mask': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }


class CAMUS(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, prompt='click',
                 plane=False):

        self.data_path = data_path
        self.name_list = os.listdir(data_path)
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return 2*len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        """Get the images"""
        if index >= len(self.name_list):
            name = self.name_list[index-len(self.name_list)]
            img_path = os.path.join(self.data_path, name, name+'_2CH_ED.nii.gz')
            msk_path = os.path.join(self.data_path, name, name+'_2CH_ED_gt.nii.gz')
            name = name+'_2CH_ED'
        else:
            name = self.name_list[index]
            img_path = os.path.join(self.data_path, name, name + '_4CH_ED.nii.gz')
            msk_path = os.path.join(self.data_path, name, name + '_4CH_ED_gt.nii.gz')
            name = name + '_4CH_ED'

        img = Image.fromarray(nib.load(img_path).get_fdata()).convert('RGB')
        mask = nib.load(msk_path).get_fdata()
        mask[mask != 1] = 0
        mask[mask>0]=1

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        image_meta_dict = {'filename_or_obj': name}
        return {
            'image': img,
            'mask': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }


class BUSI(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, prompt='click', plane=False):

        self.args = args
        self.data_path = data_path
        self.name_list = os.listdir(os.path.join(self.data_path,'img'))
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img = Image.open(os.path.join(self.data_path, 'img',name)).convert('RGB')
        mask = Image.open(os.path.join(self.data_path, 'mask',name.split('.')[0]+'_mask.png')).convert('L')

        mask = np.array(mask)
        mask[mask==mask.min()]=0
        mask[mask>0] = 255

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform(mask).int()

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'mask': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


class DLtrack(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', plane=False):

        self.args = args
        self.data_path = data_path
        if self.args.task == 'apo':
            self.image_path = os.path.join(self.data_path,'apo_images')
            self.mask_path = os.path.join(self.data_path, 'apo_masks')
        if self.args.task == 'fas':
            self.image_path = os.path.join(self.data_path, 'fasc_images_S')
            self.mask_path = os.path.join(self.data_path, 'fasc_masks_S')
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk
        self.name_list = os.listdir(self.image_path)

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img = Image.open(os.path.join(self.image_path,name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path,name)).convert('L')

        mask = np.array(mask)
        mask[mask==mask.min()] = 0
        mask[mask>0] = 1
        mask = ndimage.binary_dilation(mask, structure=np.ones((6,6)))
        mask = (mask*255).astype(np.uint8)



        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = Image.fromarray(mask)
            mask = self.transform_msk(mask).int()

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'mask': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }
