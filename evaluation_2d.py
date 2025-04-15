import os
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import cfg
import func_2d.function as function
from conf import settings
from sam_lora_image_encoder import LoRA_Sam
# from models.discriminatorlayer import discriminator
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

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    net = LoRA_Sam(net, 16)
    net.cuda()
    # optimisation
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # example of REFUGE dataset
    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform=transform_train, mode='Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform=transform_test, mode='Test')

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
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2,
                                       pin_memory=True)
        nice_support_loader = DataLoader(refuge_train_dataset, batch_size=1, sampler=support_sampler, num_workers=2,
                                         pin_memory=False)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2,
                                      pin_memory=True)
        nice_val_loader = DataLoader(refuge_test_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                     pin_memory=True)
        '''end'''

    if args.dataset == 'STARE':
        '''REFUGE data'''
        stare_train_dataset = STARE(args, args.data_path, transform=transform_train)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.05 * dataset_size))
        split_test = int(np.floor(0.25 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        test_sampler = SubsetRandomSampler(indices[split_support:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2,
                                       pin_memory=True)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=1, sampler=support_sampler, num_workers=2,
                                         pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2,
                                      pin_memory=True)
        '''end'''

    if args.dataset == 'Pandental':
        '''Pandental data'''
        stare_train_dataset = Pendal(args, args.data_path, transform=transform_train)

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
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=split_support, sampler=support_sampler,
                                         num_workers=2,
                                         pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2,
                                      pin_memory=True)
        nice_val_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                     pin_memory=True)

    if args.dataset == 'WBC':
        '''REFUGE data'''
        wbc_train_dataset = WBC(args, args.data_path, transform=transform_train, mode='Training')
        wbc_test_dataset = WBC(args, args.data_path, transform=transform_test, mode='Testing')

        dataset_size = len(wbc_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.02 * dataset_size))
        split_val = int(np.floor(0.1 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_val:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support:split_val])
        nice_train_loader = DataLoader(wbc_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2,
                                       pin_memory=True)
        nice_support_loader = DataLoader(wbc_train_dataset, batch_size=split_support, sampler=support_sampler,
                                         num_workers=2, pin_memory=False)
        nice_test_loader = DataLoader(wbc_test_dataset, batch_size=args.b, shuffle=False, num_workers=2,
                                      pin_memory=True)
        nice_val_loader = DataLoader(wbc_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                     pin_memory=True)
        '''end'''

    if args.dataset == 'CAMUS':
        '''Pandental data'''
        stare_train_dataset = CAMUS(args, args.data_path, transform=transform_train)

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
        stare_train_dataset = BUSI(args, args.data_path, transform=transform_train)

        dataset_size = len(stare_train_dataset)
        indices = list(range(dataset_size))
        split_support = int(np.floor(0.04 * dataset_size))
        split_val = int(np.floor(0.1 * dataset_size)) + split_support
        split_test = int(np.floor(0.2 * dataset_size)) + split_support
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split_test:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices[split_support: split_val])
        test_sampler = SubsetRandomSampler(indices[split_val:split_test])
        nice_train_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=2,
                                       pin_memory=False)
        nice_support_loader = DataLoader(stare_train_dataset, batch_size=split_support, sampler=support_sampler,
                                         num_workers=2, pin_memory=False)
        nice_val_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=val_sampler, num_workers=2,
                                     pin_memory=False)
        nice_test_loader = DataLoader(stare_train_dataset, batch_size=args.b, sampler=test_sampler, num_workers=2,
                                      pin_memory=False)
        '''end'''

    '''checkpoint path and tensorboard'''
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

    epoch = 0
    net.train()
    time_start = time.time()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, epoch, writer)
    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)
    net.eval()
    tol, (eiou, edice) = evaluate_sam(args, nice_val_loader, nice_support_loader, epoch, net, writer)
    logger.info(f'Val score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, epoch, net, writer)
    tol, (eiou, edice) = evaluate_sam(args, nice_train_loader, nice_support_loader, epoch, net, writer)

    epoch = 1
    net.train()
    time_start = time.time()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, epoch, writer)
    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)
    net.eval()
    tol, (eiou, edice) = evaluate_sam(args, nice_val_loader, nice_support_loader, epoch, net, writer)
    logger.info(f'Val score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, epoch, net, writer)
    tol, (eiou, edice) = evaluate_sam(args, nice_train_loader, nice_support_loader, epoch, net, writer)

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

        net.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    # validation
    epoch=99
    net.eval()

    tol, (eiou, edice) = evaluate_sam(args, nice_val_loader, nice_support_loader, epoch, net, writer)
    logger.info(f'Val score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, epoch, net, writer)
    tol, (eiou, edice) = evaluate_sam(args, nice_train_loader, nice_support_loader, epoch, net, writer)


def evaluate_sam(args, val_loader, support_loader, epoch, net: nn.Module, clean_dir=True):
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # eval mode
    net.eval()
    GPUdevice = torch.device('cuda', args.gpu_device)
    pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mask_type = torch.float32
    n_val = len(val_loader)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    # feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            # support_imgs = imgs[0:1].to(dtype=mask_type, device=GPUdevice)
            # imgs = imgs[1:].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            # support_masks = masks[0:1].to(dtype=mask_type, device=GPUdevice)
            # masks = masks[1:].to(dtype=mask_type, device=GPUdevice)
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'].to(dtype=mask_type, device=GPUdevice)
            support_masks = support_data['mask'].to(dtype=mask_type, device=GPUdevice)

            '''test'''
            with torch.no_grad():

                """ image encoder """
                backbone_out = net.sam.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                support_backbone_out = net.sam.forward_image(support_imgs)
                _, support_vision_feats, support_vision_pos_embeds, _ = net.sam._prepare_backbone_features(
                    support_backbone_out)

                '''support mask encoded with support features'''
                supportmem_features, supportmem_pos_enc = net.sam._encode_new_memory(
                    current_vision_feats=support_vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=support_masks,
                    is_mask_from_pts=True)
                supportmem_features = supportmem_features.to(torch.bfloat16)
                supportmem_pos_enc = supportmem_pos_enc[0].to(torch.bfloat16)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                vision_feats_temp = F.normalize(vision_feats_temp.reshape(B, -1))
                support_vision_feats_temp = support_vision_feats[-1].permute(1, 0, 2).reshape(
                    support_vision_feats[-1].size(1), -1, 64, 64)
                support_vision_feats_temp = F.normalize(
                    support_vision_feats_temp.reshape(support_vision_feats[-1].size(1), -1))
                similarity_scores = F.softmax(torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(), dim=1)
                support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)
                '''memory attention on support embeddings and input image feats'''
                memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2)
                # memory = (supportmem_features).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)  # [4096, Support_size, 64]
                memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2)
                # memory = memory.reshape(-1,memory.size(2)).unsqueeze(1).repeat(1,B,1) # [4096*Support_size, Batch_size, 64]
                memory = memory.reshape(-1, memory.size(2), memory.size(3))
                # memory_pos_enc = memory_pos_enc.reshape(-1, memory.size(2)).unsqueeze(1).repeat(1, B, 1)
                memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2), memory_pos_enc.size(3))

                vision_feats[-1] = net.sam.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos_enc,
                    num_obj_ptr_tokens=0
                )
                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                """ prompt encoder """
                se, de = net.sam.sam_prompt_encoder(
                    points=None,  # (coords_torch, labels_torch)
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )
                # y_0 = net.sam.sam_mask_decoder(
                #     image_embeddings=image_embed,
                #     image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                #     sparse_prompt_embeddings = se,
                #     dense_prompt_embeddings = de,
                #     multimask_output=False, # args.multimask_output if you want multiple masks
                #     repeat_image=False,  # the image is already batched
                #     high_res_features = high_res_feats)[0]
                # y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))
                # box_corordinates = random_box_gai(y_0_pred).cuda()
                # se, de = net.sam.sam_prompt_encoder(
                #         points=None, #(coords_torch, labels_torch)
                #         boxes=box_corordinates,
                #         masks=None,
                #         batch_size=B,
                #     )

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                # binary mask and calculate loss, iou, dice
                total_loss += lossfunc(pred, masks)
                pred = (pred > 0.5).float()
                temp = eval_seg(pred, masks, threshold)
                total_eiou += temp[0]
                total_dice += temp[1]

                '''vis images'''
                # if ind % args.vis == 0:
                namecat = 'Test'
                for i in range(len(name)):
                    raw_imgs = imgs[i].permute(1,2,0).cpu().numpy()
                    nii_prediction = nib.Nifti1Image(raw_imgs[:,:,0], affine=np.eye(4))
                    nib.save(nii_prediction,
                             os.path.join('/home/yxing2/CAMUS', name[i]+"_img.nii.gz"))
                    raw_imgs = masks[i].permute(1, 2, 0).cpu().numpy()
                    nii_prediction = nib.Nifti1Image(raw_imgs[:, :, 0], affine=np.eye(4))
                    nib.save(nii_prediction,
                             os.path.join('/home/yxing2/CAMUS', name[i]+"_mask2.nii.gz"))
                    raw_imgs = pred[i].permute(1, 2, 0).cpu().numpy()
                    nii_prediction = nib.Nifti1Image(raw_imgs[:, :, 0], affine=np.eye(4))
                    nib.save(nii_prediction,
                             os.path.join('/home/yxing2/CAMUS', name[i]+"_"+str(epoch)+"_prediction2.nii.gz"))

            pbar.update()

    return total_loss / n_val, tuple([total_eiou / n_val, total_dice / n_val])

if __name__ == '__main__':
    main()
