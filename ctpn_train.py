
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import csv
import glob
import cv2
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from ctpn_utils import cal_rpn,shrink_poly
import config
from config import IMAGE_MEAN


# 数据加载器
def readxml(path):
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))

                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes), imgfile

# for ctpn text detection
class VOCDataset(Dataset):
    def __init__(self,
                 datadir,
                 labelsdir):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        print(img_path)
        xml_path = os.path.join(self.labelsdir, img_name.replace('.jpg', '.xml'))
        gtbox, _ = readxml(xml_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


def readtxt(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


class ICDARDataset(Dataset):
    def __init__(self,
                 datadir):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))

        self.datadir = datadir
        self.img_names = glob.glob(self.datadir+'/*.jpg')
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = img_name
        print(img_path)

        img = cv2.imread(img_path)
        h, w, c = img.shape
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        txt_path = img_name.replace('.jpg', '.txt')
        gtbox, _ = readtxt(txt_path)
        gtbox[:,:,0] = gtbox[:,:,0] * 1024/w
        gtbox[:,:,1] = gtbox[:,:,1] * 1024/h
        h, w, c = img.shape

        res_polys = []
        for poly in gtbox:
            # delete polys with width less than 10 pixel
            if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                continue

            res = shrink_poly(poly)
            res = res.reshape([-1, 4, 2])
            for r in res:
                x_min = np.min(r[:, 0])
                y_min = np.min(r[:, 1])
                x_max = np.max(r[:, 0])
                y_max = np.max(r[:, 1])

                res_polys.append([x_min, y_min, x_max, y_max])

        gtbox = np.asarray(res_polys)

        # x_min = gtbox[:, :, 0].min(axis=-1)
        # x_max = gtbox[:, :, 0].max(axis=-1)
        # y_min = gtbox[:, :, 1].min(axis=-1)
        # y_max = gtbox[:, :, 1].max(axis=-1)
        # gtbox = np.asarray([x_min,y_min,x_max,y_max]).T

        if True:
            for p in gtbox:
                cv2.rectangle(img, (p[0],p[1]),(p[2],p[3]), color=(0, 0, 255), thickness=2)
            fig, axs = plt.subplots(1, 1, figsize=(30, 30))
            axs.imshow(img[:, :, ::-1])
            axs.set_xticks([])
            axs.set_yticks([])
            plt.tight_layout()
            plt.show()
            plt.close()

        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    imgs,clss,regrs = [],[],[]
    for i,c,r in batch:
        imgs.append(i)
        clss.append(c)
        regrs.append(r)

    imgs = torch.stack(imgs, 0)
    clss = torch.stack(clss, 0)
    regrs = torch.stack(regrs, 0)

    return imgs,clss,regrs


# 模型 =============================
class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        '''
        try:
            cls = target[:, :, 0]
            regr = target[:, :, 1:3]
            regr_keep = (cls == 1).nonzero()
            regr_true = regr[np.asarray(regr_keep.cpu()).T]
            regr_pred = input[np.asarray(regr_keep.cpu()).T]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff<1.0/self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device)


class RPN_CLS_Loss(nn.Module):
    def __init__(self,device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        # y_true = target[0][0]
        # cls_keep = (y_true != -1).nonzero()[:, 0]
        # cls_true = y_true[cls_keep].long()
        # cls_pred = input[0][cls_keep]
        # loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)  # original is sparse_softmax_cross_entropy_with_logits
        # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8

        y_t = target[:,0,:]
        c_k = (y_t != -1).nonzero()
        c_t = y_t[np.asarray(c_k.cpu()).T].long()
        c_p = input[np.asarray(c_k.cpu()).T]
        loss = F.nll_loss(F.log_softmax(c_p, dim=-1), c_t)

        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        return loss.to(self.device)


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = BasicConv(512, 512, 3,1,1,bn=False)
        self.brnn = nn.GRU(512,128, bidirectional=True, batch_first=True)
        self.lstm_fc = BasicConv(256, 512,1,1,relu=True, bn=False)
        self.rpn_class = BasicConv(512, 10*2, 1, 1, relu=False,bn=False)
        self.rpn_regress = BasicConv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        # rpn
        x = self.rpn(x)

        x1 = x.permute(0,2,3,1).contiguous()  # channels last
        b = x1.size()  # batch_size, h, w, c
        x1 = x1.view(b[0]*b[1], b[2], b[3])

        x2, _ = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256])

        x3 = x3.permute(0,3,1,2).contiguous()  # channels first
        x3 = self.lstm_fc(x3)
        x = x3

        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)

        cls = cls.permute(0,2,3,1).contiguous()
        regr = regr.permute(0,2,3,1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)
        regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2)

        return cls, regr


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
    parser.add_argument('--image-dir', type=str, default=config.image_dir)
    parser.add_argument('--labels-dir', type=str, default=config.xml_dir)
    parser.add_argument('--pretrained-weights', type=str,default=pre_weights)
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

    # dataset = VOCDataset(args['image_dir'], args['labels_dir'])
    dataset = ICDARDataset(args['image_dir'])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=args['num_workers'],collate_fn=collate)
    model = CTPN_Model()
    model.to(device)
    
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        if checkpoints_weight == './checkpoints/ctpn-end.pth':
            cc = torch.load(checkpoints_weight, map_location=device)
            for i,j in zip(list(model.state_dict().keys())[:26],list(cc.keys())[:26]):
                model.state_dict().get(i).copy_(cc.get(j))
        else:
            cc = torch.load(checkpoints_weight, map_location=device)
            model.load_state_dict(cc['model_state_dict'])
            resume_epoch = cc['epoch']

    params_to_uodate = model.parameters()
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)
    
    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(resume_epoch+1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#'*50)
        epoch_size = len(dataset) // 1
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)
    
        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()
    
            out_cls, out_regr = model(imgs)
            loss_cls = critetion_cls(out_cls, clss)
            loss_regr = critetion_regr(out_regr, regrs)
    
            loss = loss_cls + loss_regr  # total loss
            loss.backward()
            optimizer.step()
    
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i+1

            if batch_i % 50 == 0:
                print(f'Ep:{epoch}/{epochs-1}--'
                      f'Batch:{batch_i}/{epoch_size}\n'
                      f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
                      f'Epoch: loss_cls:{epoch_loss_cls/mmp:.4f}--loss_regr:{epoch_loss_regr/mmp:.4f}--'
                      f'loss:{epoch_loss/mmp:.4f}\n')
    
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

