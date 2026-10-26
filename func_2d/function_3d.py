import logging
import math
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance

from func_2d.utils import vis_image
from func_3d.utils import generate_box_new

torch.backends.cudnn.benchmark = True


def train_sam(args, net: nn.Module, optimizer, train_loader, support_loader, epoch):
    if torch.cuda.get_device_properties(args.gpu_device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device('cuda', args.gpu_device)
    net.sam.sam_mask_decoder.dynamic_multimask_via_stability = False
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    image_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)
    image_std = torch.tensor(
        [0.229, 0.224, 0.225], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)
    pos_weight = torch.tensor([2.0], device=GPUdevice)
    net.train()
    lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epoch_loss = 0.0
    n_volumes = 0
    alpha = 0.5 * (0.95 ** epoch)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
        for pack in train_loader:
            imgs_input = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks_input = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            B, length_3d = imgs_input.shape[:2]
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'].flatten(0, 1).to(dtype=torch.float32, device=GPUdevice)
            support_masks = support_data['mask'].flatten(0, 1).to(dtype=torch.float32, device=GPUdevice)
            memory_bank_feats = []
            memory_bank_pos_enc = []

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Normalize only encoder inputs; retain raw images for visualization.
                support_backbone_out = net.sam.forward_image((support_imgs - image_mean) / image_std)
                _, support_vision_feats, _, _ = net.sam._prepare_backbone_features(support_backbone_out)
                support_probs = F.interpolate(
                    support_masks.float(), size=(args.image_size, args.image_size), mode="nearest"
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
                support_vision_feats_temp = F.normalize(
                    support_vision_feats[-1].permute(1, 0, 2).reshape(support_imgs.size(0), -1)
                )

                loss = 0
                slice_indices = (
                    range(length_3d - 1, -1, -1)
                    if getattr(args, 'reverse_propagation', False)
                    else range(length_3d)
                )
                for i in slice_indices:
                    imgs = imgs_input[:, i, ...]
                    masks = masks_input[:, i, ...]
                    backbone_out = net.sam.forward_image((imgs - image_mean) / image_std)
                    _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)
                    vision_feats_temp = F.normalize(vision_feats[-1].permute(1, 0, 2).reshape(B, -1))
                    similarity_scores = F.softmax(
                        torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(), dim=1
                    )
                    support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)
                    support_memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2)
                    support_memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2)

                    if memory_bank_feats:
                        memory_3d = torch.stack(memory_bank_feats, dim=0).flatten(3).permute(0, 3, 1, 2)
                        memory_pos_3d = torch.stack(memory_bank_pos_enc, dim=0).flatten(3).permute(0, 3, 1, 2)
                        memory = torch.cat([support_memory, memory_3d], dim=0)
                        memory_pos_enc = torch.cat([support_memory_pos_enc, memory_pos_3d], dim=0)
                    else:
                        memory = support_memory
                        memory_pos_enc = support_memory_pos_enc
                    memory = memory.reshape(-1, B, memory.size(-1))
                    memory_pos_enc = memory_pos_enc.reshape(-1, B, memory_pos_enc.size(-1))
                    vision_feats[-1] = net.sam.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos_enc,
                        num_obj_ptr_tokens=0,
                    )
                    feats = [feat.permute(1, 2, 0).reshape(B, -1, *feat_size)
                             for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                    image_embed = feats[-1]
                    high_res_feats = feats[:-1]

                    # Both passes use the original SAM2 decoder.
                    se, de = net.sam.sam_prompt_encoder(points=None, boxes=None, masks=None, batch_size=B)
                    coarse_low_res = net.sam.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )[0]
                    coarse_logits = F.interpolate(coarse_low_res, size=(args.out_size, args.out_size))
                    box_coordinates = generate_box_new((coarse_logits.detach() > 0.0).float()).to(GPUdevice)
                    se, de = net.sam.sam_prompt_encoder(points=None, boxes=box_coordinates, masks=None, batch_size=B)
                    final_low_res = net.sam.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )[0]
                    final_logits = F.interpolate(final_low_res, size=(args.out_size, args.out_size))
                    # Predicted memory input is already logits, not thresholded probabilities.
                    high_res_logits = F.interpolate(
                        final_low_res, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False
                    )

                    loss_slice = (
                        alpha * lossfunc(coarse_logits.float(), masks)
                        + (1.0 - alpha) * lossfunc(final_logits.float(), masks)
                    )
                    loss = loss + loss_slice
                    maskmem_features, maskmem_pos_enc = net.sam._encode_new_memory(
                        current_vision_feats=vision_feats,
                        feat_sizes=feat_sizes,
                        pred_masks_high_res=high_res_logits,
                        is_mask_from_pts=False,
                    )
                    # Keep temporal gradients during training; validation runs under no_grad.
                    if len(memory_bank_feats) == args.memory_bank_size:
                        memory_bank_feats.pop(0)
                        memory_bank_pos_enc.pop(0)
                    memory_bank_feats.append(maskmem_features.to(torch.bfloat16))
                    memory_bank_pos_enc.append(maskmem_pos_enc[0].to(torch.bfloat16))

                loss = loss / length_3d

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B
            n_volumes += B
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update()

    return epoch_loss / n_volumes


def validation_sam(args, val_loader, support_loader, epoch, net: nn.Module,
                   clean_dir=True, export_dir=None):
    if torch.cuda.get_device_properties(args.gpu_device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device('cuda', args.gpu_device)
    net.sam.sam_mask_decoder.dynamic_multimask_via_stability = False
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    image_mean = torch.tensor(
        [0.485, 0.456, 0.406], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)
    image_std = torch.tensor(
        [0.229, 0.224, 0.225], device=GPUdevice, dtype=torch.float32
    ).view(1, 3, 1, 1)
    pos_weight = torch.tensor([2.0], device=GPUdevice)
    net.eval()
    lossfunc = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    total_loss = 0.0
    total_eiou = 0.0
    total_dice = 0.0
    total_hd95 = 0.0
    hd95_included = 0
    hd95_excluded = 0
    n_volumes = 0
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)

    with tqdm(total=len(val_loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgs_input = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks_input = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            B, length_3d = imgs_input.shape[:2]
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'].flatten(0, 1).to(dtype=torch.float32, device=GPUdevice)
            support_masks = support_data['mask'].flatten(0, 1).to(dtype=torch.float32, device=GPUdevice)
            memory_bank_feats = []
            memory_bank_pos_enc = []
            name = pack['image_meta_dict']['filename_or_obj']
            intersection = torch.zeros(B, dtype=torch.int64, device=GPUdevice)
            predicted_count = torch.zeros_like(intersection)
            target_count = torch.zeros_like(intersection)
            volume_loss = torch.zeros(B, dtype=torch.float64, device=GPUdevice)
            volume_predictions = [None] * length_3d

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Normalize only encoder inputs; retain raw images for visualization.
                support_backbone_out = net.sam.forward_image((support_imgs - image_mean) / image_std)
                _, support_vision_feats, _, _ = net.sam._prepare_backbone_features(support_backbone_out)
                support_probs = F.interpolate(
                    support_masks.float(), size=(args.image_size, args.image_size), mode="nearest"
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
                support_vision_feats_temp = F.normalize(
                    support_vision_feats[-1].permute(1, 0, 2).reshape(support_imgs.size(0), -1)
                )

                slice_indices = (
                    range(length_3d - 1, -1, -1)
                    if getattr(args, 'reverse_propagation', False)
                    else range(length_3d)
                )
                for i in slice_indices:
                    imgs = imgs_input[:, i, ...]
                    masks = masks_input[:, i, ...]
                    backbone_out = net.sam.forward_image((imgs - image_mean) / image_std)
                    _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)
                    vision_feats_temp = F.normalize(vision_feats[-1].permute(1, 0, 2).reshape(B, -1))
                    similarity_scores = F.softmax(
                        torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(), dim=1
                    )
                    support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)
                    support_memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2)
                    support_memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2)

                    if memory_bank_feats:
                        memory_3d = torch.stack(memory_bank_feats, dim=0).flatten(3).permute(0, 3, 1, 2)
                        memory_pos_3d = torch.stack(memory_bank_pos_enc, dim=0).flatten(3).permute(0, 3, 1, 2)
                        memory = torch.cat([support_memory, memory_3d], dim=0)
                        memory_pos_enc = torch.cat([support_memory_pos_enc, memory_pos_3d], dim=0)
                    else:
                        memory = support_memory
                        memory_pos_enc = support_memory_pos_enc
                    memory = memory.reshape(-1, B, memory.size(-1))
                    memory_pos_enc = memory_pos_enc.reshape(-1, B, memory_pos_enc.size(-1))
                    vision_feats[-1] = net.sam.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos_enc,
                        num_obj_ptr_tokens=0,
                    )
                    feats = [feat.permute(1, 2, 0).reshape(B, -1, *feat_size)
                             for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                    image_embed = feats[-1]
                    high_res_feats = feats[:-1]

                    # Both passes use the original SAM2 decoder.
                    se, de = net.sam.sam_prompt_encoder(points=None, boxes=None, masks=None, batch_size=B)
                    coarse_low_res = net.sam.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )[0]
                    coarse_logits = F.interpolate(coarse_low_res, size=(args.out_size, args.out_size))
                    box_coordinates = generate_box_new((coarse_logits.detach() > 0.0).float()).to(GPUdevice)
                    se, de = net.sam.sam_prompt_encoder(points=None, boxes=box_coordinates, masks=None, batch_size=B)
                    final_low_res = net.sam.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )[0]
                    final_logits = F.interpolate(final_low_res, size=(args.out_size, args.out_size))
                    # Predicted memory input is already logits, not thresholded probabilities.
                    high_res_logits = F.interpolate(
                        final_low_res, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False
                    )

                    volume_loss += lossfunc(final_logits.float(), masks).flatten(1).mean(1)
                    binary_pred = final_logits > 0.0
                    binary_target = masks >= 0.5
                    intersection += (binary_pred & binary_target).flatten(1).sum(1)
                    predicted_count += binary_pred.flatten(1).sum(1)
                    target_count += binary_target.flatten(1).sum(1)
                    volume_predictions[i] = binary_pred.detach().cpu()
                    maskmem_features, maskmem_pos_enc = net.sam._encode_new_memory(
                        current_vision_feats=vision_feats,
                        feat_sizes=feat_sizes,
                        pred_masks_high_res=high_res_logits,
                        is_mask_from_pts=False,
                    )
                    # Keep temporal gradients during training; validation runs under no_grad.
                    if len(memory_bank_feats) == args.memory_bank_size:
                        memory_bank_feats.pop(0)
                        memory_bank_pos_enc.pop(0)
                    memory_bank_feats.append(maskmem_features.to(torch.bfloat16))
                    memory_bank_pos_enc.append(maskmem_pos_enc[0].to(torch.bfloat16))

                    if ((ind + 1) % args.vis == 0) and ((i + 1) % args.vis == 0):
                        namecat = 'Test'
                        for na in name:
                            namecat += na + '+slice' + str(i)
                        vis_image(
                            imgs, binary_pred.float(), masks,
                            os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                            reverse=False, points=None,
                        )

            intersection = intersection.double()
            predicted_count = predicted_count.double()
            target_count = target_count.double()
            volume_iou = (intersection + 1e-6) / (predicted_count + target_count - intersection + 1e-6)
            volume_dice = (2 * intersection + 1e-4) / (predicted_count + target_count + 1e-4)
            total_loss += (volume_loss / length_3d).sum().item()
            total_eiou += volume_iou.sum().item()
            total_dice += volume_dice.sum().item()
            n_volumes += B

            predictions = torch.stack(volume_predictions, dim=1)
            del volume_predictions
            for batch_index in range(B):
                prediction = predictions[batch_index, :, 0].unsqueeze(0).unsqueeze(0)
                target = (pack['mask'][batch_index, :, 0].detach().cpu() >= 0.5).unsqueeze(0).unsqueeze(0)
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

            if export_dir is not None:
                for batch_index in range(B):
                    case_id = os.path.basename(str(name[batch_index]))
                    prefix = f'{case_id}_{args.task}_epoch{epoch}_grid1024'
                    prediction = predictions[batch_index, :, 0].permute(1, 2, 0).numpy().astype(np.uint8)
                    reference = (masks_input[batch_index, :, 0] >= 0.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    for suffix, volume in (('prediction', prediction), ('reference', reference)):
                        nii = nib.Nifti1Image(volume, affine=np.eye(4))
                        nii.header['descrip'] = 'Preprocessed 1024 grid; original scan geometry unavailable'
                        nib.save(nii, os.path.join(export_dir, prefix + '_' + suffix + '.nii.gz'))
            del predictions
            pbar.update()

    mean_hd95 = total_hd95 / hd95_included if hd95_included else float('nan')
    logging.getLogger().info(
        f'HD95 (3D, evaluation-grid units, finite mean): {mean_hd95:.4f} | '
        f'included: {hd95_included}/{n_volumes} | excluded: {hd95_excluded}/{n_volumes} | epoch: {epoch}'
    )
    return total_loss / n_volumes, (total_eiou / n_volumes, total_dice / n_volumes)
