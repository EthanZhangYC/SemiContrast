#!/bin/bash



CUDA_VISIBLE_DEVICES=0,1 \
python main_coseg_ours.py  \
--batch_size 2 \
--epochs 150 \
--mode block \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth 
--gpu 0,1

# add ce and dice loss
CUDA_VISIBLE_DEVICES=3 \
python main_coseg_ours.py  \
--exp_dir results/1206_bs1_stride_1gpu_SupconConDiceCE \
--batch_size 1 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
--gpu 3








--exp_dir results/0108_bs8_DiceCE_normaltransform_bnall_noise_nolradjust \
--exp_dir results/0108_bs4_DiceCE_normaltransform_noise_nolradjust \
--exp_dir results/0109_bs4_DiceCE_normaltransform_nolradjust \
--exp_dir results/0109_bs4_DiceCE_nolradjust \

CUDA_VISIBLE_DEVICES=2,3 \
python main_cedice_ours.py  \
--exp_dir results/0109_bs4_DiceCE_normaltransform_nolradjust \
--batch_size 4 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
--gpu 2,3


CUDA_VISIBLE_DEVICES=0,1 \
python main_cedice_ours.py  \
--exp_dir results/0114_bs4_DiceCE_normaltransform_nolradjust_simclrbest_cosegstridepretrain \
--batch_size 4 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1215_bs1_stride_SupconCon_1gpu_simclrbestpretrain/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \
--gpu 0,1

CUDA_VISIBLE_DEVICES=0,1 \
python main_cedice_ours.py  \
--exp_dir results/0117_bs4_DiceCE_newtransform_nolradjust_simclrbest_cosegstridepretrain \
--batch_size 4 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1215_bs1_stride_SupconCon_1gpu_simclrbestpretrain/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \
--new_transform \
--gpu 0,1

--exp_dir results/0110_bs4_DiceCE_normaltransform_nolradjust_simclrfinal_cosegstridepretrain \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1215_bs1_stride_SupconCon_1gpu_simclrfinalpretrain/models/SupCon_mmwhs_adam_fold_1_lr_1e-05_decay_0.0001_bsz_1_temp_0.1_train_1.0_stride_stride_4/ckpt.pth \




CUDA_VISIBLE_DEVICES=2,3 \
python main_cedice_ours.py  \
--exp_dir results/0114_bs4_DiceCE_normaltransform_nolradjust_simclrbestpretrain \
--batch_size 4 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/best_model.pth \
--gpu 2,3

CUDA_VISIBLE_DEVICES=0,1 \
python main_cedice_ours.py  \
--exp_dir results/0117_bs4_DiceCE_newtransform_nolradjust_simclrbestpretrain \
--batch_size 4 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/best_model.pth \
--new_transform \
--gpu 0,1

--exp_dir results/0110_bs4_DiceCE_normaltransform_nolradjust_simclrfinalpretrain \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \








CUDA_VISIBLE_DEVICES=3 \
python main_coseg_ours.py  \
--exp_dir results/1215_bs1_stride_SupconCon_1gpu_simclrfinalpretrain \
--batch_size 1 \
--epochs 100 \
--mode stride \
--fold 1 \
--save_freq 10 \
--print_freq 500 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \
--gpu 3

--exp_dir results/1215_bs1_stride_SupconCon_1gpu_simclrbestpretrain \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/best_model.pth \

--exp_dir results/1215_bs1_block_SupconCon_1gpu_simclrfinalpretrain \
--mode block \

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
