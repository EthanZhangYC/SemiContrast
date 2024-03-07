#!/bin/bash



--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \

--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1215_bs1_stride_SupconCon_1gpu_simclrfinalpretrain/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \

--new_transform \

--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \


CUDA_VISIBLE_DEVICES=2,3 \
python main_cedice_ours.py  \
--exp_dir results/0122_bs8_DiceCE_normaltransform_nolradjust_cdsfinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/experiments/mr2ct/0119_fakemr_CDS_sgd_lr3e3_wd5e4_momentum09/checkpoints/checkpoint.pth.tar \
--gpu 2,3

CUDA_VISIBLE_DEVICES=3 \
python main_cedice_ours.py  \
--exp_dir results/0125_bs8_DiceCE_normaltransform_nolradjust_simclrfinalpretrain_1 \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \
--gpu 3














CUDA_VISIBLE_DEVICES=0 \
python main_coseg_ours.py  \
--exp_dir results/0119_bs1_stride_Con_1gpu_lr000001_wd1e4_simclrfinalpretrain_nonormalize \
--batch_size 1 \
--epochs 70 \
--mode stride \
--fold 1 \
--save_freq 10 \
--print_freq 500 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \
--gpu 0

CUDA_VISIBLE_DEVICES=1 \
python main_coseg_ours.py  \
--exp_dir results/0119_bs1_stride_SupconCon_1gpu_lr000001_wd1e4_simclrfinalpretrain_nonormalize \
--batch_size 1 \
--epochs 70 \
--mode stride \
--fold 1 \
--save_freq 10 \
--print_freq 500 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \
--gpu 1

CUDA_VISIBLE_DEVICES=0,1 \
python main_cedice_ours.py  \
--exp_dir results/0123_bs8_DiceCE_normaltransform_nolradjust_coseglocalfinalpretrain_1 \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/0119_bs1_stride_Con_1gpu_lr000001_wd1e4_simclrfinalpretrain_nonormalize/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \
--gpu 2,3

CUDA_VISIBLE_DEVICES=2,3 \
CUDA_VISIBLE_DEVICES=0,1 \
python main_cedice_ours.py  \
--exp_dir results/0306_bs8_DiceCE_normaltransform_nolradjust_cosegsemifinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/0119_bs1_stride_SupconCon_1gpu_lr000001_wd1e4_simclrfinalpretrain_nonormalize/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \
--gpu 0,1






CUDA_VISIBLE_DEVICES=2,3 \
python main_simclr.py \
--exp_dir results/0115_simclr_bs16 \
--batch_size 32 \
-e 200 

CUDA_VISIBLE_DEVICES=0 \
python main_simclr.py \
--exp_dir results/1210_simclr_bs16_ourpretrain \
--batch_size 16 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
-e 100 












CUDA_VISIBLE_DEVICES=3 \
python main_cedice_ours.py  \
--exp_dir results/0306_bs8_DiceCE_normaltransform_nolradjust_cdsfinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home1/yichen/SemiContrast/results/0119_fakemr_CDS_sgd_lr3e3_wd5e4_momentum09/checkpoints/checkpoint.pth.tar \
--gpu 3

CUDA_VISIBLE_DEVICES=3 \
python main_cedice_ours.py  \
--exp_dir results/0306_bs8_DiceCE_normaltransform_nolradjust_simclrfinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home1/yichen/SemiContrast/results/1210_simclr_bs16/final_model.pth \
--gpu 3




CUDA_VISIBLE_DEVICES=0 \
python main_cedice_ours.py  \
--exp_dir results/0306_bs8_DiceCE_normaltransform_nolradjust_coseglocalfinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home1/yichen/SemiContrast/results/0119_bs1_stride_Con_1gpu_lr000001_wd1e4_simclrfinalpretrain_nonormalize/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \
--gpu 0

CUDA_VISIBLE_DEVICES=0 \
python main_cedice_ours.py  \
--exp_dir results/0306_bs8_DiceCE_normaltransform_nolradjust_cosegsemifinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home1/yichen/SemiContrast/results/0119_bs1_stride_SupconCon_1gpu_lr000001_wd1e4_simclrfinalpretrain_nonormalize/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \
--gpu 0