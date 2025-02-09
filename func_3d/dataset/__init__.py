from .btcv import BTCV
from .amos import AMOS
from .pet_tumor import PETCT
from .acdc import ACDC
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch



def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'petct':
        petct_train_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                  prompt=args.prompt)
        petct_test_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                 prompt=args.prompt)
        dataset_size = len(petct_train_dataset)
        indices = list(range(dataset_size))
        split_support = 1
        test_dataset_size = len(petct_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5*test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        train_sampler = SubsetRandomSampler(indices[split_support:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(petct_train_dataset, batch_size=1, sampler=support_sampler, num_workers=8, pin_memory=False)
        nice_train_loader = DataLoader(petct_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=False)
        nice_test_loader = DataLoader(petct_test_dataset, batch_size=1, sampler=test_sampler, num_workers=8, pin_memory=False)
        nice_val_loader = DataLoader(petct_test_dataset, batch_size=1, sampler=val_sampler, num_workers=8,
                                     pin_memory=False)
    elif args.dataset == 'petct_distributed':
        petct_train_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                  prompt=args.prompt)
        petct_test_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                 prompt=args.prompt)
        dataset_size = len(petct_train_dataset)
        indices = list(range(dataset_size)) # train indice
        split_support = 1 # support size
        test_dataset_size = len(petct_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5*test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        #train_sampler = DistributedSampler(torch.utils.data.Subset(petct_train_dataset, indices[split_support:]))
        train_sampler = DistributedSampler(dataset=petct_train_dataset, shuffle=True)
        train_sampler.set_epoch(args.epoch)
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        #test_sampler = DistributedSampler(torch.utils.data.Subset(petct_test_dataset, indices[:split_val]))
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(petct_train_dataset, batch_size=1, sampler=support_sampler, num_workers=8, pin_memory=False)
        nice_train_loader = DataLoader(petct_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=False)
        nice_test_loader = DataLoader(petct_test_dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=False)
        nice_val_loader = DataLoader(petct_test_dataset, batch_size=args.b, sampler=val_sampler, num_workers=8,
                                     pin_memory=False)

    elif args.dataset == 'acdc':
        acdc_train_dataset = ACDC(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                    prompt=args.prompt)
        acdc_test_dataset = ACDC(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                   prompt=args.prompt)
        dataset_size = len(acdc_train_dataset)
        indices = list(range(dataset_size))
        split_support = 5
        test_dataset_size = len(acdc_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5 * test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        train_sampler = SubsetRandomSampler(indices[split_support:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(acdc_train_dataset, batch_size=1, sampler=support_sampler, num_workers=8,
                                         pin_memory=False)
        nice_train_loader = DataLoader(acdc_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8,
                                       pin_memory=False)
        nice_test_loader = DataLoader(acdc_test_dataset, batch_size=1, sampler=test_sampler, num_workers=8,
                                      pin_memory=False)
        nice_val_loader = DataLoader(acdc_test_dataset, batch_size=1, sampler=val_sampler, num_workers=8,
                                     pin_memory=False)
    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader
