#!/bin/bash


source activate py37

batch=4
dataset=Hippocampus
fold=3
head=mlp
mode=stride
temp=0.1
train_sample=1
# dataset=mmwhs


python main_coseg.py  --batch_size ${batch} --dataset ${dataset} \
    --data_folder ./data \
    --learning_rate 0.0001 \
    --epochs 60 \
    --head ${head} \
    --mode ${mode} \
    --fold ${fold} \
    --save_freq 1 \
    --print_freq 10 \
    --temp ${temp} \
    --train_sample ${train_sample} \
    --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \
    # --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \


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
CUDA_VISIBLE_DEVICES=0,1,2 \
python main_coseg_ours.py  \
--exp_dir results/test \
--batch_size 2 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
--gpu 0,1,2

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

CUDA_VISIBLE_DEVICES=1 \
python main_coseg_ours.py  \
--exp_dir results/1211_bs1_stride_1gpu_ConDiceCE \
--batch_size 1 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
--gpu 1


CUDA_VISIBLE_DEVICES=2 \
python main_cedice_ours.py  \
--exp_dir results/1215_bs8_DiceCE_normaltransform_1gpu_nbatch \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
--gpu 2

CUDA_VISIBLE_DEVICES=1,3 \
python main_cedice_ours.py  \
--exp_dir results/1215_bs8_DiceCE_normaltransform_2gpu_nbatch \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
--gpu 1,3


CUDA_VISIBLE_DEVICES=2,3 \
python main_cedice_ours.py  \
--exp_dir results/1211_bs8_stride_DiceCE_normaltransform_simclrfinalpretrain \
--batch_size 8 \
--epochs 150 \
--mode stride \
--fold 1 \
--save_freq 50 \
--print_freq 100 \
--temp 0.1 \
--resume /home/ziyuan/yichen/semi_cotrast_seg/results/1210_simclr_bs16/final_model.pth \
--gpu 2,3






CUDA_VISIBLE_DEVICES=1 \
python main_simclr.py \
--exp_dir results/1210_simclr_bs16 \
--batch_size 16 \
-e 100 

CUDA_VISIBLE_DEVICES=0 \
python main_simclr.py \
--exp_dir results/1210_simclr_bs16_ourpretrain \
--batch_size 16 \
--resume /home/ziyuan/yichen/ProtoUDA/pcs/exps/mr2mr_4labels.pth \
-e 100 
