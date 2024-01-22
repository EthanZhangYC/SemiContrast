from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import save_model
from util import get_gpu_memory_map
# from networks.unet_con import SupConUnet, LocalConUnet2, LocalConUnet3
from loss_functions.supcon_loss import SupConSegLoss, LocalConLoss, BlockConLoss
# from datasets.two_dim.NumpyDataLoader import NumpyDataSet

import torchutils
from datautils import WHS_dataset, WHS_dataset_multiview, create_loader
from networks.unet_ours import OUNet,CosineClassifier
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import medpy.metric.binary as mmb
import pdb
import logging

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='logger.info frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='where to find the pretrained model')
    parser.add_argument('--head', type=str, default="cls",
                        help='head mode, cls or mlp')
    parser.add_argument('--stride', type=int, default=4,
                        help='number of stride when doing downsampling')
    parser.add_argument('--block_size', type=int, default=4,
                        help='number of stride when doing downsampling')
    parser.add_argument('--mode', type=str, default="stride",
                        help='how to downsample the feature maps, stride or block')


    # optimization
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimization method')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='mmwhs',
                        help='dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help="path to the stored checkpoint")
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--split_dir', type=str, default=None, help='path to split pickle file')
    parser.add_argument('--fold', type=int, default=0, help='parameter for splits')
    parser.add_argument('--train_sample', type=float, default=1.0, help='parameter for sampling rate of training set')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    parser.add_argument('--gpu', type=str, default='1',
                        help='id for recording multiple runs')
    parser.add_argument('--exp_dir', type=str, default='test',
                        help='id for recording multiple runs')
    
    parser.add_argument('--new_transform', action='store_true')


    opt = parser.parse_args()
    
        
    # os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = 'data'
    else:
        opt.data_folder = os.path.join(opt.data_folder, opt.dataset, 'preprocessed')

    if opt.split_dir is None:
        opt.split_dir = os.path.join('./data', opt.dataset)
    # opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    # opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)
    opt.model_path = opt.exp_dir + '/models'
    opt.tb_path = opt.exp_dir + '/tb_log'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_fold_{}_lr_{}_decay_{}_bsz_{}_temp_{}_train_{}_{}'.\
        format(opt.method, opt.dataset, opt.optimizer, opt.fold, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.train_sample, opt.mode)

    if opt.mode =="stride":
        opt.model_name = '{}_stride_{}'.format(opt.model_name, opt.stride)
    elif opt.mode == "block":
        opt.model_name = '{}_block_{}'.format(opt.model_name, opt.block_size)

    if opt.pretrained_model_path is not None:
        opt.model_name = '{}_pretrained'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def set_loader(args):
    num_workers = 4
    batch_size = args.batch_size

    data_path_train_unl={
        'source':"/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like/org_mr",
        'target':"/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like/fake_mr"
    }
    
    # train_dataset_source = WHS_dataset_multiview(data_path_train_unl["source"], multi_view=True)
    # train_dataset_target = WHS_dataset_multiview(data_path_train_unl["target"], multi_view=True)
    # train_lbd_dataset_source = WHS_dataset_multiview("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_labeled/datalist/org_4labeled.txt", multi_view=False, labeled=True, transforms_label=False, mode='supcon')
    
    # train_dataset_source = WHS_dataset(data_path_train_unl["source"], transforms=True)
    # train_dataset_target = WHS_dataset(data_path_train_unl["target"], transforms=True)
    if args.new_transform:
        train_lbd_dataset_source = WHS_dataset("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_labeled/datalist/org_4labeled.txt", labeled=True, new_transforms=True, mode='train')
    else:
        train_lbd_dataset_source = WHS_dataset("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_labeled/datalist/org_4labeled.txt", labeled=True, mode='train')
    # train_loader_source = create_loader(
    #     train_dataset_source,
    #     batch_size,
    #     num_workers=num_workers,
    #     shuffle=True
    # )
    # train_loader_target = create_loader(
    #     train_dataset_target,
    #     batch_size,
    #     num_workers=num_workers,
    #     shuffle=True
    # )
    train_lbd_loader_source = create_loader(
        train_lbd_dataset_source,
        batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    if args.new_transform:
        real_test_unl_dataset_target = WHS_dataset("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like_test/fake_mr", labeled=True, is_real_test=True, new_transforms=True, mode='test')
    else:
        real_test_unl_dataset_target = WHS_dataset("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like_test/fake_mr", labeled=True, is_real_test=True, mode='test')
    real_test_unl_loader_target = create_loader(
        real_test_unl_dataset_target,
        1,
        num_workers=num_workers,
        shuffle=False
    )
    
    # num_batches = max(len(train_loader_source), len(train_loader_target)) + 1
    num_batches = len(train_lbd_loader_source) + 1
    
    train_lbd_iter_source = ForeverDataIterator(train_lbd_loader_source)
    # train_iter_source = ForeverDataIterator(train_loader_source)
    # train_iter_target = ForeverDataIterator(train_loader_target)
    # # train_iter_source = train_iter_target = None
    train_iter_source=train_iter_target=None
    
    train_loader = (train_lbd_iter_source, train_iter_source, train_iter_target, num_batches)
    
    return train_loader, real_test_unl_loader_target



def set_model(args, logger):
    model = OUNet(in_chns=1, class_num=64, out_dim=64).cuda()
        
    if args.resume is not None:
        logger.info(f"Loading pretrained model '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if 'mr2mr_4labels' in args.resume:
            del checkpoint['decoder.out_conv.weight']
            del checkpoint['decoder.out_conv.bias']
        model_state_dict = checkpoint
        model.load_state_dict(model_state_dict, strict=False)

    # classification head
    cls_head = CosineClassifier(num_class=5, inc=64, temp=0.1).cuda()
    torchutils.weights_init(cls_head)

    # device_ids=list(map(int, args.gpu.split(',')))
    # if len(args.gpu)>1 and torch.cuda.is_available():
    #     device_id = []
    #     for i in range((len(args.gpu) + 1) // 2):
    #         device_id.append(i)
    #     net = torch.nn.DataParallel(net, device_ids=device_id)


    if len(args.gpu)>1:
        device_ids = []
        for i in range((len(args.gpu) + 1) // 2):
            device_ids.append(i)
        print('nn parallel', device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)
        cls_head = nn.DataParallel(cls_head, device_ids=device_ids)
    else:
        model = model.cuda()
        cls_head = cls_head.cuda()
    
    # # criterion = nn.CrossEntropyLoss().cuda()
    # # criterion_noreduce = nn.CrossEntropyLoss(reduce=False).cuda()
    # if args.mode == "block":
    #     criterion = BlockConLoss(temperature=args.temp, block_size=args.block_size)
    # elif args.mode == "stride":
    #     criterion = LocalConLoss(temperature=args.temp, stride=args.stride)
    # else:
    #     raise NotImplementedError("The feature downsampling mode is not supported yet!")
    criterion=None
    
    return model, cls_head, criterion

         

def set_optimizer(args, model, cls_head):
    lr = 0.01
    weight_decay = 5e-4
    conv_lr_ratio = 0.1

    parameters = []
    # batch_norm layer: no weight_decay
    params_bn, _ = torchutils.split_params_by_name(model, "bn")
    parameters.append({"params": params_bn, "weight_decay": 0.0})
    assert len(params_bn)==0
    # conv layer: small lr
    _, params_conv = torchutils.split_params_by_name(model, ["fc", "bn"])
    if conv_lr_ratio:
        parameters[0]["lr"] = lr * conv_lr_ratio
        parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
    else:
        parameters.append({"params": params_conv})
    # fc layer
    params_fc, _ = torchutils.split_params_by_name(model, "fc")
    params_fc.extend(list(cls_head.parameters()))
    parameters.append({"params": params_fc})

    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    return optimizer



def train(train_loader, model, cls_head, criterion, logger_tb, optimizer, epoch, opt, dice_loss, ce_loss, logger):
    """one epoch training"""
    model.train()
    cls_head.train()
    
    train_lbd_iter_source, train_iter_source, train_iter_target, num_batches = train_loader

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    # for idx, data_batch in enumerate(train_loader):
    # num_batches = len(self.train_lbd_loader_source) + 1
    for idx in range(num_batches):
        # if idx>5:
        #     break
        
        data_time.update(time.time() - end)
        # # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        
        # superivsed setting
        _, images_lbd, labels = next(train_lbd_iter_source)
        images_lbd,labels = images_lbd.float().cuda(), labels.cuda()
                
        # noise = torch.clamp(torch.randn_like(images_lbd) * 0.1, -0.2, 0.2)
        # feat_lbd,_,_,_ = model(images_lbd+noise)
        feat_lbd,_,_,_ = model(images_lbd)
        
        feat_lbd = F.normalize(feat_lbd, dim=1)
        
        out_lbd = cls_head(feat_lbd.permute(0,2,3,1).reshape(-1,64))
        out_lbd = out_lbd.view(-1,256,256,5).permute(0,3,1,2)
        
        loss_dice, _ = dice_loss(out_lbd, labels)
        loss_ce = ce_loss(out_lbd, labels.long())
        
        
        # _, images_unl_source, _ = next(train_iter_source)
        # _, images_unl_target, _ = next(train_iter_target)       
        # images_unl = torch.cat([images_unl_source,images_unl_target], dim=0)
        # images_unl = images_unl.float().cuda()
        # _,_,_,_ = model(images_unl)
        

        loss = loss_dice + loss_ce
        
        losses.update(loss.item(), 256)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # features = model(imgs)

        # loss_time = time.time() - start
        # logger.info(loss_time)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info info
        if (idx + 1) % opt.print_freq == 0:
            num_iteration = idx + 1 + (epoch-1)*len(train_loader)
            logger_tb.add_scalar("train_loss", losses.avg, num_iteration)
            logger.info('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, num_batches, batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


@torch.no_grad()
def test(args, test_loader, model, cls_head, logger):
    model.eval()
    # if mean_teacher:
    #     model_t = model_t.eval()
    # # Domain Adaptation
    # if cls:
    cls_head.eval()

    # current_val_metric = test_score(use_teacher=False)
    current_val_metric = test_score(args, test_loader, model, cls_head, logger)
    current_val_metric_lcc = test_score_lcc(args, test_loader, model, cls_head, logger)
    
    return current_val_metric, current_val_metric_lcc

@torch.no_grad()
def test_score(args, testloader, model, cls_head, logger):
    logger.info('-------testing non-lcc with Teacher:-------')

    # testloader = real_test_unl_loader_target
    dice_list = []
    assd_list = []

    vol_lst = ['1003','1008','1014','1019']
    pred_vol_lst = [np.zeros((256,256,256)) for _ in range(4)]
    label_vol_lst = [np.zeros((256,256,256)) for _ in range(4)]

    for i_batch, (indices, images, labels, vol_indices, slice_indices) in enumerate(testloader):
        volume_batch = images.float().cuda()
        label_batch = labels.numpy()

        vol_id = vol_lst.index(vol_indices[0])
        slice_id = int(slice_indices[0])

        # if mean_teacher and use_teacher:
        #     outputs,_,_,_ = model_t(volume_batch)
        # else:
        outputs,_,_,_ = model(volume_batch)
        
        outputs = F.normalize(outputs, dim=1)  
        outputs = cls_head(outputs.permute(0,2,3,1).reshape(-1,64)) 
        outputs = outputs.view(-1,256,256,5).permute(0,3,1,2)
        
        outputs = torch.softmax(outputs, dim = 1)
        outputs = torch.argmax(outputs, dim = 1).cpu().numpy()

        pred_vol_lst[vol_id][slice_id] = outputs[0,...]
        label_vol_lst[vol_id][slice_id] = label_batch[0,...]
        
    for i in range(4):
        for c in range(1, 5):

            pred_test_data_tr = pred_vol_lst[i].copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label_vol_lst[i].copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
            try:
                assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
            except:
                assd_list.append(np.nan)
                
    dice_arr = 100 * np.reshape(dice_list, [-1, 4]).transpose()


    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)
    
    assd_arr = np.reshape(assd_list, [-1, 4]).transpose()

    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)

    logger.info('Dice:')
    logger.info('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
    logger.info('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    logger.info('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
    logger.info('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    logger.info('Mean:%.1f\n' % np.mean(dice_mean))

    logger.info('ASSD:')
    logger.info('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
    logger.info('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    logger.info('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
    logger.info('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    logger.info('Mean:%.1f' % np.mean(assd_mean))
    logger.info('--------------')

    return np.mean(dice_mean)

@torch.no_grad()
def test_score_lcc(args, testloader, model, cls_head, logger):
    logger.info('-------testing lcc with Student:-------')

    dice_list = []
    assd_list = []
    vol_lst = ['1003','1008','1014','1019']
    pred_vol_lst = [np.zeros((256,256,256)) for _ in range(4)]
    label_vol_lst = [np.zeros((256,256,256)) for _ in range(4)]
    
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0:
            dice = mmb.dc(pred, gt)
            assd = mmb.assd(pred, gt)
            return dice, assd
        else:
            return 0, 0

    for i_batch, (indices, images, labels, vol_indices, slice_indices) in enumerate(testloader):
        volume_batch = images.float().cuda()
        label_batch = labels.numpy()
        vol_id = vol_lst.index(vol_indices[0])
        slice_id = int(slice_indices[0])

        outputs,_,_,_ = model(volume_batch)        
        outputs = F.normalize(outputs, dim=1)  
        outputs = cls_head(outputs.permute(0,2,3,1).reshape(-1,64)) 
        outputs = outputs.view(-1,256,256,5).permute(0,3,1,2)
        
        outputs = torch.softmax(outputs, dim = 1)
        # outputs = torch.argmax(outputs, dim = 1).cpu().numpy()
        outputs = torch.argmax(outputs, dim = 1).squeeze(0).cpu().numpy()

        # pred_vol_lst[vol_id][slice_id] = outputs[0,...]
        # label_vol_lst[vol_id][slice_id] = label_batch[0,...]
        pred_vol_lst[vol_id][slice_id] = outputs
        label_vol_lst[vol_id][slice_id] = label_batch


    for vol_id in range(4):
        metric_list = []
        prediction = torchutils.post_process(pred_vol_lst[vol_id])
        for i in range(1, 5):
            # if (label_batch[0,...] == i).sum()>0:
            metric_list.append(calculate_metric_percase(prediction == i, label_vol_lst[vol_id] == i))
        dice_list.append([d for d, asd in metric_list])
        assd_list.append([asd for d, asd in metric_list])
        # else:
        #     for c in range(1, num_class):
        #         pred_test_data_tr = pred_vol_lst[i-1].copy()
        #         pred_test_data_tr[pred_test_data_tr != c] = 0

        #         pred_gt_data_tr = label_vol_lst[i].copy()
        #         pred_gt_data_tr[pred_gt_data_tr != c] = 0

        #         dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
        #         try:
        #             assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))
        #         except:
        #             assd_list.append(np.nan)
    
    dice_arr = 100 * np.reshape(np.array(dice_list), [-1, 4]).transpose()
    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)
    
    assd_arr = np.reshape(assd_list, [-1, 4]).transpose()
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)

    logger.info('Dice:')
    logger.info('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
    logger.info('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    logger.info('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
    logger.info('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    logger.info('Mean:%.1f\n' % np.mean(dice_mean))
    
    logger.info('ASSD:')
    logger.info('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
    logger.info('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    logger.info('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
    logger.info('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    logger.info('Mean:%.1f' % np.mean(assd_mean))
    logger.info('--------------')
    
    return np.mean(dice_mean)



def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class DiceLoss(nn.Module):
    def __init__(self, n_classes, softmax = True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        if target.dim() < inputs.dim():
            target = target.unsqueeze(1)
        if target.shape[1] == 1 and target.shape[1] < inputs.shape[1]:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict & target shape do not match, with inputs={inputs.shape}, target={target.shape})'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes, np.array(class_wise_dice)
    
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    opt = parse_option()
    
    if not os.path.exists(opt.exp_dir):
        os.mkdir(opt.exp_dir)
    logger = get_logger(opt.exp_dir+'/log')


    # build data loader
    # train_loader = set_loader(opt)
    train_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, cls_head, criterion = set_model(opt, logger)
    dice_loss = DiceLoss(n_classes=5)
    ce_loss = nn.CrossEntropyLoss()

    # build optimizer
    optimizer = set_optimizer(opt, model, cls_head)

    # tensorboard
    logger_tb = SummaryWriter(opt.tb_folder)

    # training routine
    best_val_metric = 0.
    best_val_metric_lcc = 0.
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, cls_head, criterion, logger_tb, optimizer, epoch, opt, dice_loss, ce_loss, logger)
        
        if epoch%2==0:
            current_val_metric,current_val_metric_lcc = test(opt, test_loader, model, cls_head, logger)
            # update information
            if current_val_metric >= best_val_metric:
                best_val_metric = current_val_metric
                best_val_epoch = epoch        
            if current_val_metric_lcc >= best_val_metric_lcc:
                best_val_metric_lcc = current_val_metric_lcc
                best_val_epoch_lcc = epoch
            logger.info('Best Dice: %f (%d), lcc: %f (%d)'%(best_val_metric,best_val_epoch,best_val_metric_lcc,best_val_epoch_lcc))
            
        time2 = time.time()
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # tensorboard logger
        logger_tb.add_scalar('loss', loss, epoch)
        logger_tb.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt.pth'.format(epoch=epoch))
            save_model(model, cls_head, optimizer, opt, epoch, save_file)

    # save the last model
    # save_file = os.path.join(
    #   opt.save_folder, 'last.pth')
    save_model(model, cls_head, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    print(torch.cuda.device_count())
    main()
