<h1 align="center">● SAM2-SAP: Automatic Prompt Driven SAM2 for Medical Image Segmentation</h1>


SAM2-SAP is a customized segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address both 2D and 3D medical image segmentation tasks without requring manual prompts. This method is elaborated based on the paper [SAM2-SAP: Automatic Prompt Driven SAM2 for Medical Image Segmentation].

##  Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate sam2_sap``

 You can download SAM2 checkpoint from checkpoints folder:
 
 ``bash download_ckpts.sh``

 Further Note: We tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Ubuntu 22.04
Conda Version: 24.3.0
Python Version: 3.12.4
Torch Version: 12.4.1
```
Pretrained weight will be released soon

 ## Example Cases
 
 ### 2D case - CAMUS Ultrasound Segmentation

**Step1:** Run the training by:

``python train_2d.py -net sam2 -exp_name WBC_Abl_Support16 -vis 999 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 256 -out_size 256 -b 8 -val_freq 5 -dataset WBC -data_path /data_path/data/WBC/ -support_size 1 -lr 1e-3``

 **Step2:** Run the validation by:
 
``python evaluation_2d.py -net sam2 -exp_name WBC_Abl_Support16 -vis 999 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 256 -out_size 256 -b 8 -val_freq 5 -dataset WBC -data_path /data_path/data/WBC/ -support_size 1 -lr 1e-3``

 ### 3D case - Amos22 Multi-organ CT&MRI Segmentation
 
 **Step1:** Run the training by:

``python train_3d.py -net sam2 -exp_name AMOS_float32 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -val_freq 3 -prompt bbox -dataset amos -data_path /data_path/data/amos22/MRI/ -video_length 10 -lr 1e-2 -vis 99 -b 2 -support_size 4 -task left+kidney``

**Step2:** Run the validation by:


 ``python evaluate_3d.py -net sam2 -exp_name AMOS_float32 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -val_freq 3 -prompt bbox -dataset amos -data_path /data_path/data/amos22/MRI/ -video_length 10 -lr 1e-2 -vis 99 -b 2 -support_size 4 -task left+kidney``
 
 ### multi-gpu training, Using PET/CT dataset
 
 
 ``python -m torch.distributed.run --nproc_per_node=2 train_3d_dpp.py -disted True -net sam2 -exp_name PETCT_dpp -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 256 -val_freq 5 -prompt bbox -dataset petct_distributed -data_path /data_path/seg_data -image_size 256 -out_size 256 -video_length 5 -lr 1e-4``
 
 
 ### To use different resolutions, change the configurations in sam2_hiera_s.yaml


