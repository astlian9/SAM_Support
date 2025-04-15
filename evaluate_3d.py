import os
import time
from tqdm import tqdm
import nibabel as nib
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import cfg
import func_2d.function_3d as function
from conf import settings
from sam_lora_image_encoder import LoRA_Sam
# from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *
from func_3d.dataset import get_dataloader
from sam2_train.build_sam import build_sam2

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

    sam_net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    net = LoRA_Sam(sam_net, 16)
    net.to(device=GPUdevice)

    '''load data'''
    nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader = get_dataloader(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    '''train one epoch'''
    net.train()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, 0)
    print(loss)
    net.eval()
    tol, (eiou, edice) = evaluate_sam(args, nice_val_loader, nice_support_loader, 0, net)
    # tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, 13, net, None)
    print(f'Val score: {tol}, IOU: {eiou}, DICE: {edice}.')
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, 0, net)
    # tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, 13, net, None)
    print(f'Test score: {tol}, IOU: {eiou}, DICE: {edice}.')
    '''second epoch'''
    net.train()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, 1)
    print(loss)
    net.eval()
    tol, (eiou, edice) = evaluate_sam(args, nice_val_loader, nice_support_loader, 1, net)
    # tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, 13, net, None)
    print(f'Val score: {tol}, IOU: {eiou}, DICE: {edice}.')
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, 1, net)
    # tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, 13, net, None)
    print(f'Test score: {tol}, IOU: {eiou}, DICE: {edice}.')



    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        # start_epoch = checkpoint['epoch']
        # best_tol = checkpoint['best_tol']
        print('best_dice=',checkpoint['best_dice'],'at epoch', checkpoint['epoch'])
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
        args.path_helper = checkpoint['path_helper']
        net.to(device=GPUdevice)
        print(f'=> loaded checkpoint {checkpoint_file} ')


    net.eval()
    tol, (eiou, edice) =evaluate_sam(args, nice_val_loader, nice_support_loader, 99, net)
    # tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, 13, net, None)
    print(f'Val score: {tol}, IOU: {eiou}, DICE: {edice}.')
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, 99, net)
    # tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, 13, net, None)
    print(f'Test score: {tol}, IOU: {eiou}, DICE: {edice}.')


def evaluate_sam(args, val_loader, support_loader, epoch, net: nn.Module, clean_dir=True):

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True
    # eval mode
    net.eval()

    GPUdevice = torch.device('cuda',args.gpu_device)
    pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mask_type = torch.float32

    n_val = len(val_loader)
    threshold = (0.5, 0.5, 0.5, 0.5, 0.5)

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    # feat_sizes = [(256, 256), (128, 128), (64, 64)]
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0
    flag = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            # input image and gt masks
            imgs_input = pack['image'].to(dtype=mask_type, device=GPUdevice)
            # support_imgs = imgs[0:1].to(dtype = mask_type, device = GPUdevice)
            # imgs = imgs[1:].to(dtype = mask_type, device = GPUdevice)
            masks_input = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            # support_masks = masks[0:1].to(dtype = mask_type, device = GPUdevice)
            # masks = masks[1:].to(dtype = mask_type, device = GPUdevice)
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'][0].to(dtype=mask_type, device=GPUdevice)
            support_masks = support_data['mask'][0].to(dtype=mask_type, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj'][0]
            # print(imgs_input.shape, masks_input.shape)

            '''generate support embeddings'''
            support_backbone_out = net.sam.forward_image(support_imgs)
            _, support_vision_feats, support_vision_pos_embeds, _ = net.sam._prepare_backbone_features(
                support_backbone_out)
            '''encode support mask into support embeddings'''
            high_res_supportmasks = F.interpolate(support_masks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            supportmem_features, supportmem_pos_enc = net.sam._encode_new_memory(
                current_vision_feats=support_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_supportmasks,
                is_mask_from_pts=True)
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
            supportmem_features = supportmem_features.to(dtype=torch.bfloat16)
            supportmem_features = supportmem_features.to(device=GPUdevice,
                                                         non_blocking=True)  # [Support_set, 64, 64, 64]
            supportmem_pos_enc = supportmem_pos_enc[0].to(dtype=torch.bfloat16)  # [Support_set, 64, 64, 64]
            supportmem_pos_enc = supportmem_pos_enc.to(device=GPUdevice, non_blocking=True)
            support_vision_feats_temp = support_vision_feats[-1].permute(1, 0, 2).reshape(
                support_vision_feats[-1].size(1), -1, 64, 64)
            support_vision_feats_temp = F.normalize(
                support_vision_feats_temp.reshape(support_vision_feats[-1].size(1), -1)).cuda()
            length_3d = imgs_input.size(1)
            memory_bank_feats = []
            memory_bank_pos_enc = []
            final_predictions = []
            if epoch==0:
                raw_imgs = imgs_input[0].permute(2,3,0,1).cpu().numpy()
                # nii_prediction = nib.Nifti1Image(raw_imgs, affine=np.eye(4))
                # nib.save(nii_prediction, os.path.join('/home/yxing2/ACDC', name + "_img.nii.gz"))
                raw_masks = masks_input[0].permute(2,3,0,1).cpu().numpy()
                raw_masks[raw_masks>0]=1
                nii_prediction = nib.Nifti1Image(raw_masks, affine=np.eye(4))
                nib.save(nii_prediction, os.path.join('/home/yxing2/AMOS', name[0] + "_mask_"+args.task+"_"+str(epoch)+".nii.gz"))

            '''test'''
            with torch.no_grad():
                for i in range(length_3d):
                    imgs = imgs_input[:, i, ...]
                    masks = masks_input[:, i, ...]

                    '''Train image encoder'''
                    backbone_out = net.sam.forward_image(imgs)
                    _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)
                    # dimension hint for your future use
                    # vision_feats: list: length = 3
                    # vision_feats[0]: torch.Size([65536, batch, 32])
                    # vision_feats[1]: torch.Size([16384, batch, 64])
                    # vision_feats[2]: torch.Size([4096, batch, 256])
                    # vision_pos_embeds[0]: torch.Size([65536, batch, 256])
                    # vision_pos_embeds[1]: torch.Size([16384, batch, 256])
                    # vision_pos_embeds[2]: torch.Size([4096, batch, 256])
                    '''Train memory attention to condition on support image'''
                    B = vision_feats[-1].size(1)  # batch size
                    # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
                    # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
                    # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed
                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64).cuda()
                    vision_feats_temp = F.normalize(vision_feats_temp.reshape(B, -1))
                    similarity_scores = F.softmax(torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(),
                                                  dim=1).cuda()
                    support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)

                    '''memory attention on support embeddings and input image feats'''
                    support_memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2).cuda(
                        non_blocking=True)  # [4096, Support_size,batch, 64]
                    support_memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2).cuda(
                        non_blocking=True)
                    # memory = memory.reshape(-1, memory.size(2), memory.size(3))  # [4096*Support_size, Batch_size, 64]
                    # memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2), memory_pos_enc.size(3))

                    '''memory attention on past frames'''
                    if len(memory_bank_feats) == 0:
                        memory = support_memory.reshape(-1, support_memory.size(2),
                                                        support_memory.size(3))  # [4096*Support_size, Batch_size, 64]
                        memory_pos_enc = support_memory_pos_enc.reshape(-1, support_memory_pos_enc.size(2),
                                                                        support_memory_pos_enc.size(3))
                    else:
                        L = len(memory_bank_feats)
                        memory_stack_ori = torch.stack(memory_bank_feats, dim=0)
                        memory_pos_stack_ori = torch.stack(memory_bank_pos_enc, dim=0)
                        memory_3d = memory_stack_ori.flatten(3).permute(0, 3, 1, 2)
                        memory_pos_3d = memory_pos_stack_ori.flatten(3).permute(0, 3, 1, 2)
                        memory = torch.cat([support_memory, memory_3d], dim=0)
                        memory_pos_enc = torch.cat([support_memory_pos_enc, memory_pos_3d], dim=0)
                        memory = memory.reshape(-1, memory.size(2),
                                                        memory.size(3))  # [4096*Support_size, Batch_size, 64]
                        memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2),
                                                                        memory_pos_enc.size(3))

                    vision_feats[-1] = net.sam.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos_enc,
                        num_obj_ptr_tokens=0
                    )
                    feats = [feat.cuda(non_blocking=True).permute(1, 2, 0).view(B, -1, *feat_size)
                             for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                    image_embed = feats[-1]
                    high_res_feats = feats[:-1]


                    with torch.no_grad():
                        '''infer from cross attentioned feats and zero prompts'''
                        se, de = net.sam.sam_prompt_encoder(
                            points=None,  # (coords_torch, labels_torch)
                            boxes=None,
                            masks=None,
                            batch_size=B,
                        )
                        y_0 = net.sam.sam_mask_decoder(
                            image_embeddings=image_embed,
                            image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,  # args.multimask_output if you want multiple masks
                            repeat_image=False,  # the image is already batched
                            high_res_features=high_res_feats)[0]
                        '''generate pseudo mask'''
                        y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))
                        '''generate box prompt'''
                        box_corordinates = random_box_gai((y_0_pred > 0.5).float()).cuda()
                        se, de = net.sam.sam_prompt_encoder(
                            points=None,  # (coords_torch, labels_torch)
                            boxes=box_corordinates,
                            masks=None,
                            batch_size=B,
                        )

                    '''train mask decoder'''
                    low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,  # args.multimask_output if you want multiple masks
                        repeat_image=False,  # the image is already batched
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
                    flag = flag + 1
                    final_predictions.append(pred)

                    '''encode new mask and features to memory'''
                    maskmem_features_3d, maskmem_pos_enc_3d = net.sam._encode_new_memory(
                        current_vision_feats=vision_feats,
                        feat_sizes=feat_sizes,
                        pred_masks_high_res=high_res_multimasks,
                        is_mask_from_pts=False)
                    # dimension hint for your future use
                    # maskmem_features: torch.Size([batch, 64, 64, 64])
                    # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]

                    maskmem_features_3d = maskmem_features_3d.to(torch.bfloat16)
                    maskmem_pos_enc_3d = maskmem_pos_enc_3d[0].to(torch.bfloat16)
                    ''' store it in memory bank'''
                    if len(memory_bank_feats) < args.memory_bank_size:
                        memory_bank_feats.append(maskmem_features_3d)
                        memory_bank_pos_enc.append(maskmem_pos_enc_3d)
                    else:
                        memory_bank_feats.pop(0)
                        memory_bank_pos_enc.pop(0)
                        memory_bank_feats.append(maskmem_features_3d)
                        memory_bank_pos_enc.append(maskmem_pos_enc_3d)


            prediction = torch.cat(final_predictions, dim=0)
            prediction = prediction*255
            prediction = prediction.squeeze(1).permute(1,2,0).cpu().numpy().astype(np.uint8)
            prediction[prediction>0]=1
            nii_prediction = nib.Nifti1Image(prediction, affine=np.eye(4))
            nib.save(nii_prediction, os.path.join('/home/yxing2/AMOS', name[0] + "_prediction_"+args.task+"_"+str(epoch)+".nii.gz"))


            pbar.update()

    # nii_prediction = nib.Nifti1Image(prediction, affine=np.eye(4))
    # nib.save(nii_prediction, os.path.join('/home/yxing2/SAM2_modified/runs',name+".nii.gz"))
    n_val = n_val * length_3d
    print(length_3d, flag, n_val)
    return total_loss / flag, tuple([total_eiou / flag, total_dice / flag])



if __name__ == '__main__':
    main()
