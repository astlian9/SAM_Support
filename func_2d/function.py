
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import cfg
from conf import settings
from func_2d.utils import *
import pandas as pd


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32

torch.backends.cudnn.benchmark = True


def train_sam(args, net: nn.Module, optimizer, train_loader, support_loader, epoch, writer):
    
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    
    # train mode
    net.train()
    optimizer.zero_grad()

    # init
    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    # feat_sizes = [(256, 256), (128, 128), (64, 64)]


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            
            to_cat_memory = []
            to_cat_memory_pos = []

            # input image and gt masks
            imgs = pack['image'].to(dtype = mask_type, device = GPUdevice)
            # support_imgs = imgs[0:1].to(dtype = mask_type, device = GPUdevice)
            # imgs = imgs[1:].to(dtype = mask_type, device = GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device = GPUdevice)
            # support_masks = masks[0:1].to(dtype = mask_type, device = GPUdevice)
            # masks = masks[1:].to(dtype = mask_type, device = GPUdevice)
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'].to(dtype=mask_type, device=GPUdevice)
            support_masks = support_data['mask'].to(dtype=mask_type, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            # print(imgs.shape, masks.shape, support_imgs.shape, support_masks.shape)

            # click prompt: unsqueeze to indicate only one click, add more click across this dimension
            # if 'pt' in pack:
            #     pt_temp = pack['pt'].to(device = GPUdevice)
            #     pt = pt_temp.unsqueeze(1)
            #     point_labels_temp = pack['p_label'].to(device = GPUdevice)
            #     point_labels = point_labels_temp.unsqueeze(1)
            #     coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
            #     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            # else:
            #     coords_torch = None
            #     labels_torch = None
            # end of modification

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
            
            # if len(memory_bank_list) == 0:
            #     vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.sam.hidden_dim)).to(device="cuda")
            #     vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.sam.hidden_dim)).to(device="cuda")
            #
            # else:
            #     for element in memory_bank_list:
            #         to_cat_memory.append((element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_features
            #         to_cat_memory_pos.append((element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_pos_enc
            #     memory = torch.cat(to_cat_memory, dim=0)
            #     memory_pos = torch.cat(to_cat_memory_pos, dim=0)
            #
            #
            #     memory = memory.repeat(1, B, 1)
            #     memory_pos = memory_pos.repeat(1, B, 1)
            #
            #
            #     vision_feats[-1] = net.sam.memory_attention(
            #         curr=[vision_feats[-1]],
            #         curr_pos=[vision_pos_embeds[-1]],
            #         memory=memory,
            #         memory_pos=memory_pos,
            #         num_obj_ptr_tokens=0
            #         )

            # feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
            #          for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            #
            # image_embed = feats[-1]
            # high_res_feats = feats[:-1]
            
            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed
            '''generate support embeddings'''
            support_backbone_out = net.sam.forward_image(support_imgs)
            _, support_vision_feats, support_vision_pos_embeds, _ = net.sam._prepare_backbone_features(
                support_backbone_out)

            # support_feats = [support_feat.permute(1, 2, 0).view(B, -1, *support_feat_size)
            #                  for support_feat, support_feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

            # support_image_embed = support_feats[-1]
            # support_high_res_feats = support_feats[:-1]

            '''encode support mask into support embeddings'''
            supportmem_features, supportmem_pos_enc = net.sam._encode_new_memory(
                current_vision_feats=support_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=support_masks,
                is_mask_from_pts=True)
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]

            supportmem_features = supportmem_features.to(dtype=torch.bfloat16)
            supportmem_features = supportmem_features.to(device=GPUdevice, non_blocking=True) #[Support_set, 64, 64, 64]
            supportmem_pos_enc = supportmem_pos_enc[0].to(dtype=torch.bfloat16) # [Support_set, 64, 64, 64]
            supportmem_pos_enc = supportmem_pos_enc.to(device=GPUdevice, non_blocking=True)

            vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64).cuda()
            vision_feats_temp = F.normalize(vision_feats_temp.reshape(B, -1))
            support_vision_feats_temp = support_vision_feats[-1].permute(1, 0, 2).reshape(support_vision_feats[-1].size(1), -1, 64, 64)
            support_vision_feats_temp = F.normalize(support_vision_feats_temp.reshape(support_vision_feats[-1].size(1), -1)).cuda()
            similarity_scores = F.softmax(torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(), dim=1).cuda()
            support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)
            '''memory attention on support embeddings and input image feats'''
            memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2).cuda(non_blocking=True)  # [4096, Support_size, 64]
            memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2).cuda(non_blocking=True)
            memory = memory.reshape(-1, memory.size(2), memory.size(3)) # [4096*Support_size, Batch_size, 64]
            memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2), memory_pos_enc.size(3))
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
                    sparse_prompt_embeddings = se,
                    dense_prompt_embeddings = de,
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats)[0]

                '''generate pseudo mask'''
                y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))


                '''generate box prompt'''
                box_corordinates = random_box_gai((y_0_pred>0.5).float()).cuda()
                se, de = net.sam.sam_prompt_encoder(
                        points=None, #(coords_torch, labels_torch)
                        boxes=box_corordinates,
                        masks=None,
                        batch_size=B,
                    )
            # dimension hint for your future use
            # se: torch.Size([batch, n+1, 256])
            # de: torch.Size([batch, 256, 64, 64])



            
            '''train mask decoder'''       
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats
                )
            # dimension hint for your future use
            # low_res_multimasks: torch.Size([batch, multimask_output, 256, 256])
            # iou_predictions.shape:torch.Size([batch, multimask_output])
            # sam_output_tokens.shape:torch.Size([batch, multimask_output, 256])
            # object_score_logits.shape:torch.Size([batch, 1])
            
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
            # y_0_pred = F.interpolate(y_0,size=(args.out_size,args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)

            # backpropagation
            y_0_mask = torch.log(torch.sigmoid((y_0_pred>0.5).float())+1e-10)
            pred_mask = (pred>0.5).float()
            loss = lossfunc(pred, masks)
            + 0.005 * F.kl_div(y_0_mask, pred_mask, reduction='batchmean')
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            pbar.update()

    return epoch_loss/len(train_loader)




def validation_sam(args, val_loader, support_loader, epoch, net: nn.Module, clean_dir=True):

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # eval mode
    net.eval()

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
                y_0 = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = se,
                    dense_prompt_embeddings = de,
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats)[0]
                y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))
                box_corordinates = random_box_gai(y_0_pred).cuda()
                se, de = net.sam.sam_prompt_encoder(
                        points=None, #(coords_torch, labels_torch)
                        boxes=box_corordinates,
                        masks=None,
                        batch_size=B,
                    )

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, 
                    repeat_image=False,  
                    high_res_features = high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)

                # binary mask and calculate loss, iou, dice
                total_loss += lossfunc(pred, masks)
                pred = (pred> 0.5).float()
                temp = eval_seg(pred, masks, threshold)
                total_eiou += temp[0]
                total_dice += temp[1]

                '''vis images'''
                if ind % args.vis == 0:
                # if (epoch == 0) or (epoch==99):
                    namecat = 'Test'
                    for na in name:
                        img_name = na
                        namecat = namecat + img_name + '+'
                    vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)
                            
            pbar.update()

    return total_loss/ n_val , tuple([total_eiou/n_val, total_dice/n_val])

