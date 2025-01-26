import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

import cfg
import func_2d.function_3d as function
from conf import settings
from sam_lora_image_encoder import LoRA_Sam
# from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *
from func_3d.dataset import get_dataloader
torch.multiprocessing.set_sharing_strategy('file_system')


def process(args):
    rank = 0

    if args.disted:
        dist.init_process_group(backend="nccl", init_method="env://")
        # print(os.environ['LOCAL_RANK'])
        rank = int(os.environ['LOCAL_RANK'])
    args.device = torch.device(f"cuda:{rank}")
    args.gpu_device = int(rank)
    torch.cuda.set_device(args.device)
    GPUdevice = args.device

    # load network
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    net = LoRA_Sam(net, 16)
    net.to(args.device)

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

        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    if rank == 0:
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)

    # optimization
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if args.dist:
        net = DistributedDataParallel(net, device_ids=[args.device])

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    '''load data'''
    nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    if rank == 0:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
        # use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(settings.EPOCH):

        # training
        dist.barrier()
        args.epoch = epoch
        nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader = get_dataloader(args)
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, epoch)
        if rank==0 :
            logger.info(f'GPU id: {rank} Train loss: {loss} || @ epoch {epoch}.')
        print(f'GPU id: {rank} Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        torch.cuda.empty_cache()

        # validation
        if (epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1) & (rank == 0):
            net.eval()

            tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, epoch, net)
            logger.info(f'GPU id: {rank} Val score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            torch.cuda.empty_cache()

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()
            if rank == 0:
                # if edice  > best_dice:
                if edice > best_dice:
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
    if rank == 0:
        checkpoint_final = torch.load(os.path.join(args.path_helper['ckpt_path'], 'best_dice_checkpoint.pth'))
        net.load_state_dict(checkpoint_final['state_dict'])
        net.eval()
        tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, nice_support_loader, epoch, net, writer)
        logger.info(f'Test score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        writer.close()

    dist.destroy_process_group()


if __name__ == '__main__':
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = cfg.parse_args()

    process(args=args)

