#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: train.py.py
@time: 2019/6/25 20:00
@desc:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.squeezenet import squeezenet1_1 as Squeezenet

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
import glob
import cv2
import random
import argparse
import config
from config import IMAGE_MEAN


class Image_Dataset(Dataset):
    def __init__(self,datadir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))

        self.datadir = datadir
        self.img_names = glob.glob(self.datadir + '/*.jpg')

        M1 = cv2.getRotationMatrix2D((256 / 2, 256 / 2),90,1)
        M2 = cv2.getRotationMatrix2D((256 / 2, 256 / 2),180,1)
        M3 = cv2.getRotationMatrix2D((256 / 2, 256 / 2),270,1)
        M4 = cv2.getRotationMatrix2D((256 / 2, 256 / 2),0,1)
        self.M = [M1,M2,M3,M4]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = img_name

        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img - IMAGE_MEAN

        r = random.randint(0,3)
        img = cv2.warpAffine(img, self.M[r], (256, 256))

        # transform to torch tensor
        img = torch.from_numpy(img.transpose([2, 0, 1])).float()
        return img, r


random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

num_workers = 1
epochs = 20
lr = 1e-3
resume_epoch = 0
pre_weights = os.path.join(config.checkpoints_dir, 'ctpn-end.pth')


def get_arguments():
    parser = argparse.ArgumentParser(description='Pytorch CTPN For TexT Detection')
    parser.add_argument('--num-workers', type=int, default=num_workers)
    parser.add_argument('--image-dir', type=str, default=config.image_dir1)
    parser.add_argument('--labels-dir', type=str, default=config.xml_dir)
    parser.add_argument('--pretrained-weights', type=str, default=pre_weights)
    return parser.parse_args()


def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth.tar'):
    check_path = os.path.join(config.checkpoints_dir,
                              f'ctpn_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')

    torch.save(state, check_path)
    print('saving to {}'.format(check_path))


args = vars(get_arguments())

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoints_weight = args['pretrained_weights']
    if os.path.exists(checkpoints_weight):
        pretrained = False

    dataset = Image_Dataset(args['image_dir'])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args['num_workers'])

    model = Squeezenet(pretrained=False)

    # pretrained_dict = model0.state_dict()
    # model = Squeezenet(num_classes=4)
    # model_dict = model.state_dict()
    #
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {v[0]: v[1] for v, v1 in zip(pretrained_dict.items(), model_dict.items()) if v[1].shape == v1[1].shape}
    #
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    #
    # # 加载我们真正需要的state_dict
    # model.load_state_dict(model_dict)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1)),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.LogSoftmax(dim=1)
        )

    model.to(device)

    params_to_uodate = model.parameters()
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)

    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(resume_epoch + 1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#' * 50)
        epoch_size = len(dataset) // 1
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)

        for batch_idx, (imgs, clss) in enumerate(train_loader):
            imgs = imgs.to(device)
            clss = clss.to(device)
            optimizer.zero_grad()
            out_cls = model(imgs)
            loss = F.nll_loss(out_cls, clss)
            loss.backward()
            optimizer.step()
            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print(f'Epoch:{epoch}--{epoch_loss_cls:.4f}--{epoch_loss_regr:.4f}--{epoch_loss:.4f}')
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(),
                             'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

