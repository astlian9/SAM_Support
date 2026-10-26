
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance

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
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.train()
    epoch_loss = 0
    net.sam.sam_mask_decoder.dynamic_multimask_via_stability = False
    lossfunc = criterion_G
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    image_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)
    image_std = torch.tensor(
        [0.229, 0.224, 0.225], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'].to(dtype=mask_type, device=GPUdevice)
            support_masks = support_data['mask'].to(dtype=mask_type, device=GPUdevice)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Normalize images here: several datasets share image/mask transforms.
                backbone_out = net.sam.forward_image((imgs - image_mean) / image_std)
                _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                support_backbone_out = net.sam.forward_image((support_imgs - image_mean) / image_std)
                _, support_vision_feats, _, _ = net.sam._prepare_backbone_features(support_backbone_out)

                # The SAM2 memory API expects logits, not binary mask probabilities.
                support_probs = F.interpolate(
                    support_masks.float(), size=support_imgs.shape[-2:], mode="nearest"
                )
                support_logits = torch.logit(support_probs.clamp(1e-6, 1.0 - 1e-6))
                supportmem_features, supportmem_pos_enc = net.sam._encode_new_memory(
                    current_vision_feats=support_vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=support_logits,
                    is_mask_from_pts=False,
                )
                supportmem_features = supportmem_features.to(torch.bfloat16)
                supportmem_pos_enc = supportmem_pos_enc[0].to(torch.bfloat16)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                vision_feats_temp = F.normalize(vision_feats_temp.reshape(B, -1))
                support_vision_feats_temp = support_vision_feats[-1].permute(1, 0, 2).reshape(
                    support_vision_feats[-1].size(1), -1, 64, 64
                )
                support_vision_feats_temp = F.normalize(
                    support_vision_feats_temp.reshape(support_vision_feats[-1].size(1), -1)
                )
                similarity_scores = F.softmax(
                    torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(), dim=1
                )
                support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)

                memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2)
                memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2)
                memory = memory.reshape(-1, memory.size(2), memory.size(3))
                memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2), memory_pos_enc.size(3))
                vision_feats[-1] = net.sam.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos_enc,
                    num_obj_ptr_tokens=0,
                )
                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                # Coarse mask: the original decoder with no explicit query prompt.
                se, de = net.sam.sam_prompt_encoder(
                    points=None, boxes=None, masks=None, batch_size=B
                )
                y_0 = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )[0]
                y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))

                # Box extraction is discrete; coarse BCE supplies its own learning signal.
                box_coordinates = random_box_gai(
                    (y_0_pred.detach() > 0.0).float()
                ).to(GPUdevice)
                se, de = net.sam.sam_prompt_encoder(
                    points=None, boxes=box_coordinates, masks=None, batch_size=B
                )
                final_logits = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )[0]
                pred = F.interpolate(final_logits, size=(args.out_size, args.out_size))

                loss_weight = 0.5 * (0.95 ** epoch)
                coarse_loss = lossfunc(y_0_pred.float(), masks)
                final_loss = lossfunc(pred.float(), masks)
                loss = (1.0 - loss_weight) * final_loss + loss_weight * coarse_loss

            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            pbar.update()

    return epoch_loss / len(train_loader)


def validation_sam(args, val_loader, support_loader, epoch, net: nn.Module, clean_dir=True):
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.eval()
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    n_val = len(val_loader)
    n_images = 0
    total_loss = 0
    total_eiou = 0
    total_dice = 0
    total_hd95 = 0.0
    hd95_included = 0
    hd95_excluded = 0
    net.sam.sam_mask_decoder.dynamic_multimask_via_stability = False
    lossfunc = criterion_G
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    image_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)
    image_std = torch.tensor(
        [0.229, 0.224, 0.225], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'].to(dtype=mask_type, device=GPUdevice)
            support_masks = support_data['mask'].to(dtype=mask_type, device=GPUdevice)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Normalize images here: several datasets share image/mask transforms.
                backbone_out = net.sam.forward_image((imgs - image_mean) / image_std)
                _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                support_backbone_out = net.sam.forward_image((support_imgs - image_mean) / image_std)
                _, support_vision_feats, _, _ = net.sam._prepare_backbone_features(support_backbone_out)

                # The SAM2 memory API expects logits, not binary mask probabilities.
                support_probs = F.interpolate(
                    support_masks.float(), size=support_imgs.shape[-2:], mode="nearest"
                )
                support_logits = torch.logit(support_probs.clamp(1e-6, 1.0 - 1e-6))
                supportmem_features, supportmem_pos_enc = net.sam._encode_new_memory(
                    current_vision_feats=support_vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=support_logits,
                    is_mask_from_pts=False,
                )
                supportmem_features = supportmem_features.to(torch.bfloat16)
                supportmem_pos_enc = supportmem_pos_enc[0].to(torch.bfloat16)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                vision_feats_temp = F.normalize(vision_feats_temp.reshape(B, -1))
                support_vision_feats_temp = support_vision_feats[-1].permute(1, 0, 2).reshape(
                    support_vision_feats[-1].size(1), -1, 64, 64
                )
                support_vision_feats_temp = F.normalize(
                    support_vision_feats_temp.reshape(support_vision_feats[-1].size(1), -1)
                )
                similarity_scores = F.softmax(
                    torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(), dim=1
                )
                support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)

                memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2)
                memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2)
                memory = memory.reshape(-1, memory.size(2), memory.size(3))
                memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2), memory_pos_enc.size(3))
                vision_feats[-1] = net.sam.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos_enc,
                    num_obj_ptr_tokens=0,
                )
                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                # Coarse mask: the original decoder with no explicit query prompt.
                se, de = net.sam.sam_prompt_encoder(
                    points=None, boxes=None, masks=None, batch_size=B
                )
                y_0 = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )[0]
                y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))

                # Box extraction is discrete; coarse BCE supplies its own learning signal.
                box_coordinates = random_box_gai(
                    (y_0_pred.detach() > 0.0).float()
                ).to(GPUdevice)
                se, de = net.sam.sam_prompt_encoder(
                    points=None, boxes=box_coordinates, masks=None, batch_size=B
                )
                final_logits = net.sam.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )[0]
                pred = F.interpolate(final_logits, size=(args.out_size, args.out_size))
                total_loss += lossfunc(pred.float(), masks).item() * B

            pred = (pred > 0.0).float()
            temp = eval_seg(pred, masks, (0.5,))
            total_eiou += temp[0] * B
            total_dice += temp[1] * B
            n_images += B

            for batch_index in range(B):
                prediction = pred[batch_index:batch_index + 1].detach().cpu().bool()
                target = masks[batch_index:batch_index + 1].detach().cpu() >= 0.5
                pred_present = prediction.any().item()
                target_present = target.any().item()
                if pred_present and target_present:
                    hd95 = compute_hausdorff_distance(
                        prediction, target, include_background=True,
                        distance_metric="euclidean", percentile=95,
                        directed=False, spacing=1.0,
                    ).item()
                elif not pred_present and not target_present:
                    hd95 = 0.0
                else:
                    hd95 = float('nan')
                if math.isfinite(hd95):
                    total_hd95 += hd95
                    hd95_included += 1
                else:
                    hd95_excluded += 1
                del prediction, target

            if ind % args.vis == 0:
                namecat = 'Test'
                for na in name:
                    namecat = namecat + na + '+'
                vis_image(
                    imgs, pred, masks,
                    os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                    reverse=False, points=None,
                )
            pbar.update()

    mean_hd95 = total_hd95 / hd95_included if hd95_included else float('nan')
    logging.getLogger().info(
        f'HD95 (2D, evaluation-grid units, finite mean): {mean_hd95:.4f} | '
        f'included: {hd95_included}/{n_images} | excluded: {hd95_excluded}/{n_images} | epoch: {epoch}'
    )
    return total_loss / n_images, (
        total_eiou / n_images, total_dice / n_images
    )
