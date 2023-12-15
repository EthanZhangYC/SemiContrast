import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from loss_functions.nt_xent import NTXentLoss
import os
import shutil
import sys
import pickle

# from datasets.two_dim.NumpyDataLoader import NumpyDataSet
# from networks.unet_con import GlobalConUnet, MLP

from datautils import WHS_dataset, WHS_dataset_multiview, create_loader
from networks.unet_ours import OUNet_encoder

apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class MLP(nn.Module):
    def __init__(self, input_channels=1024, num_class=128):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        y = self.f2(y)

        return y

class SimCLR(object):

    def __init__(self, config, logger):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config['save_dir'], 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])
        self.logger = logger

        # split_dir = os.path.join(self.config["base_dir"], "splits.pkl")
        # data_dir = os.path.join(self.config["base_dir"], "preprocessed")
        # self.logger.info(data_dir)
        # with open(split_dir, "rb") as f:
        #     splits = pickle.load(f)
        # tr_keys = splits[0]['train'] + splits[0]['val'] + splits[0]['test']
        # val_keys = splits[0]['val']
        # self.train_loader = NumpyDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["batch_size"],
        #                                  keys=tr_keys, do_reshuffle=True, mode='simclr')
        # self.val_loader = NumpyDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["val_batch_size"],
        #                                  keys=val_keys, do_reshuffle=True, mode='simclr')
        
        data_path_train_unl={
            'source':"/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like/org_mr",
            'target':"/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like/fake_mr"
        }
        train_dataset = WHS_dataset_multiview([data_path_train_unl["source"],data_path_train_unl["target"]], multi_view=True, mode='simclr')
        self.train_loader = create_loader(
            train_dataset,
            self.config['batch_size'],
            num_workers=4,
            shuffle=True
        )
        val_dataset = WHS_dataset_multiview("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like_test/fake_mr", multi_view=True, mode='simclr')
        self.val_loader = create_loader(
            val_dataset,
            self.config['batch_size'],
            num_workers=4,
            shuffle=False
        )
        # val_dataset = WHS_dataset("/home/ziyuan/yichen/ProtoUDA/data/data_mmwhs/mr/mr_like_test/fake_mr", labeled=True, is_real_test=True)
        # self.val_loader = create_loader(
        #     val_dataset,
        #     1,
        #     num_workers=4,
        #     shuffle=False
        # )
        print(len(self.train_loader),len(self.val_loader))
        
        # self.model = GlobalConUnet()
        self.model = OUNet_encoder(in_chns=1, class_num=64, out_dim=64).cuda()
        if self.config['resume'] is not None:
            print(f"Loading pretrained model '{self.config['resume']}'")
            checkpoint = torch.load(self.config['resume'], map_location="cpu")
            del checkpoint['decoder.out_conv.weight']
            del checkpoint['decoder.out_conv.bias']
            model_state_dict = checkpoint
            self.model.load_state_dict(model_state_dict, strict=False)


        self.head = MLP(num_class=256)

        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        # dist.init_process_group(backend='nccl')
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
            self.head = nn.DataParallel(self.head)

        self.model.to(self.device)
        self.head.to(self.device)

        # self.model = self._load_pre_trained_weights(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, head, xis, xjs, n_iter):

        # get the representations and the projections
        ris = model(xis)  # [N,C]
        zis = head(ris)

        # get the representations and the projections
        rjs = model(xjs)  # [N,C]
        zjs = head(rjs)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
                                                                    last_epoch=-1)

        for epoch_counter in range(self.config['epochs']):
            self.logger.info("=====Training Epoch: %d =====" % epoch_counter)
            self.model.train()
            for i, (_,images_unl,_) in enumerate(self.train_loader):
                # if i>5:
                #     break
                
                self.optimizer.zero_grad()

                # xis = xis['data'][0].float().to(self.device)
                # xjs = xjs['data'][0].float().to(self.device)
                xis = images_unl[0].float().to(self.device)
                xjs = images_unl[1].float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.logger.info("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()
                self.optimizer.step()
                n_iter += 1

            self.logger.info("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(self.val_loader)
                self.logger.info("Val:[{0}] loss: {loss:.4f}".format(epoch_counter, loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config['exp_dir'],'best_model.pth'))
                    self.logger.info("saving best: %d"%(epoch_counter))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
        torch.save(self.model.state_dict(), os.path.join(self.config['exp_dir'],'final_model.pth'))

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
                
            # for (xis, xjs) in valid_loader:
            #     xis = xis['data'][0].float().to(self.device)
            #     xjs = xjs['data'][0].float().to(self.device)
            
            for i, (_,images_unl,_) in enumerate(self.val_loader):
                xis = images_unl[0].float().to(self.device)
                xjs = images_unl[1].float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss
