import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import cfg
import func_2d.function as function
from conf import settings
from sam_lora_image_encoder import LoRA_Sam
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *


def main():
    # use bfloat16 for the entire work
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)
    torch.cuda.set_device(GPUdevice)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    net = LoRA_Sam(net, 16)
    net.cuda()
    # optimisation
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


    '''segmentation data'''
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        # transforms.RandomRotation(15),  # Rotate image by ±15 degrees
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((args.out_size, args.out_size)),
        transforms.ToTensor(),
    ])

    
    # example of REFUGE dataset
    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, mode = 'Test')

        dataset_size = len(refuge_train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.125 * dataset_size))
        test_dataset_size = len(refuge_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5 * test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        train_sampler = SubsetRandomSampler(indices[split:])
        support_sampler = SubsetRandomSampler(indices[:split])
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2, pin_memory=True)
        nice_support_loader = DataLoader(refuge_train_dataset, batch_size=1,  sampler=support_sampler, num_workers=2, pin_memory=False)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2, pin_memory=True)
        nice_val_loader = DataLoader(refuge_test_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                      pin_memory=True)
        '''end'''

    if args.dataset == 'STARE':
        '''REFUGE data'''
        stare_train_dataset = STARE(args, args.data_path, transform = transform_train)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.05 * dataset_size))
        split_test = int(np.floor(0.25 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        test_sampler = SubsetRandomSampler(indices[split_support:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2, pin_memory=True)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=1,  sampler=support_sampler, num_workers=2, pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2, pin_memory=True)
        '''end'''

    if args.dataset == 'Pandental':
        '''Pandental data'''
        stare_train_dataset = Pendal(args, args.data_path, transform = transform_train)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.15 * dataset_size))
        split_val = int(np.floor(0.1 * dataset_size)) + split_support
        split_test = int(np.floor(0.2 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support: split_val])
        test_sampler = SubsetRandomSampler(indices[split_val:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2,
                                       pin_memory=True)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=split_support, sampler=support_sampler, num_workers=2,
                                         pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2,
                                      pin_memory=True)
        nice_val_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                     pin_memory=True)

    if args.dataset == 'WBC':
        '''REFUGE data'''
        wbc_train_dataset = WBC(args, args.data_path, transform = transform_train, mode = 'Training')
        wbc_test_dataset = WBC(args, args.data_path, transform = transform_test, mode = 'Testing')

        dataset_size = len(wbc_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.02 * dataset_size))
        split_val = int(np.floor(0.1 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_val:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support:split_val])
        nice_train_loader = DataLoader(wbc_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2, pin_memory=True)
        nice_support_loader = DataLoader(wbc_train_dataset, batch_size=split_support,  sampler=support_sampler, num_workers=2, pin_memory=False)
        nice_test_loader = DataLoader(wbc_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        nice_val_loader = DataLoader(wbc_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2, pin_memory=True)
        '''end'''

    if args.dataset == 'CAMUS':
        '''Pandental data'''
        stare_train_dataset = CAMUS(args, args.data_path, transform = transform_train)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.02 * dataset_size))
        split_val = int(np.floor(0.1 * dataset_size)) + split_support
        split_test = int(np.floor(0.2 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support: split_val])
        test_sampler = SubsetRandomSampler(indices[split_val:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8,
                                       pin_memory=True)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=split_support, sampler=support_sampler,
                                         num_workers=8,
                                         pin_memory=True)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=8,
                                      pin_memory=True)
        nice_val_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=8,
                                     pin_memory=True)
        '''end'''

    if args.dataset == 'BUSI':
        '''Pandental data'''
        stare_train_dataset = BUSI(args, args.data_path, transform = transform_train)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.04 * dataset_size))
        split_val = int(np.floor(0.1*dataset_size)) + split_support
        split_test = int(np.floor(0.2 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support: split_val])
        test_sampler = SubsetRandomSampler(indices[split_val:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2, pin_memory=False)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=split_support,  sampler=support_sampler, num_workers=2, pin_memory=False)
        nice_val_loader = DataLoader(stare_train_dataset, batch_size=args.b,  sampler=val_sampler, num_workers=2, pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2, pin_memory=False)
        '''end'''

    if args.dataset == 'DLtrack':
        '''Pandental data'''
        stare_train_dataset = DLtrack(args, args.data_path, transform = transform_train,transform_msk=transform_mask)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.02 * dataset_size))
        split_val = int(np.floor(0.15 * dataset_size)) + split_support
        split_test = int(np.floor(0.3 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support: split_val])
        test_sampler = SubsetRandomSampler(indices[split_val:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2,
                                       pin_memory=True)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=args.support_size, sampler=support_sampler, num_workers=2,
                                         pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2,
                                      pin_memory=True)
        nice_val_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                     pin_memory=True)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0


    for epoch in range(settings.EPOCH):

        # training
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, epoch, writer)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        # if epoch==0:
        #     tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, nice_support_loader, epoch, net,
        #                                                  writer)
            # tol, (eiou, edice) = function.validation_sam(args, nice_train_loader, nice_support_loader, epoch, net,
            #                                              writer)
        # validation
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

            tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, epoch, net, writer)
            logger.info(f'Val score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            #if edice > best_dice:
            if  edice > best_dice:
                best_dice = edice
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
            else:
                is_best = False

    checkpoint_final = torch.load(os.path.join(args.path_helper['ckpt_path'],'best_dice_checkpoint.pth'))
    net.load_state_dict(checkpoint_final['state_dict'])
    net.eval()
    epoch=99
    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, nice_support_loader, epoch, net, writer)
    logger.info(f'Test score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
    tol, (eiou, edice) = function.validation_sam(args, nice_train_loader, nice_support_loader, epoch, net, writer)
    writer.close()


if __name__ == '__main__':
    main()
