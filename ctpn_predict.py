#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: plt.py
@time: 2019/6/20 14:21
@desc:
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cv2
import numpy as np

import torch.nn as nn
import torchvision.models as models

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn_utils import resize
import config


prob_thresh = 0.7
width = 1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = os.path.join(config.checkpoints_dir, 'ctpn_ep28_0.0084_0.0142_0.0226.pth.tar')
# img_path = '/home/OCR/icdar2017/train/image_7.jpg'
img_path = '/mnt/f/Data/icdar2017rctw_train_v1.2/train/part1/image_0.jpg'

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


image = cv2.imread(img_path)
image = resize(image, width=width,height=width)
image_c = image.copy()
h, w = image.shape[:2]
image = image.astype(np.float32) - config.IMAGE_MEAN
image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()


model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'],strict=False)
model.to(device)
model.eval()

def dis(image):
    plt.imshow(image[:,:,::-1])
    plt.show()


with torch.no_grad():
    image = image.to(device)
    cls, regr = model(image)
    cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
    regr = regr.cpu().numpy()
    anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
    bbox = bbox_transfor_inv(anchor, regr)
    bbox = clip_box(bbox, [h, w])

    fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
    select_anchor = bbox[fg, :]
    select_score = cls_prob[0, fg, 1]
    select_anchor = select_anchor.astype(np.int32)

    keep_index = filter_bbox(select_anchor, 16)

    # nsm
    select_anchor = select_anchor[keep_index]
    select_score = select_score[keep_index]
    select_score = np.reshape(select_score, (select_score.shape[0], 1))
    nmsbox = np.hstack((select_anchor, select_score))
    keep = nms(nmsbox, 0.3)
    select_anchor = select_anchor[keep]
    select_score = select_score[keep]

    # text line-
    textConn = TextProposalConnectorOriented()
    text = textConn.get_text_lines(select_anchor, select_score, [h, w])
    print(text)

    for i in text:
        s = str(round(i[-1] * 100, 2)) + '%'
        i = [int(j) for j in i]
        cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
        cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
        cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
        cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
        cv2.putText(image_c, s, (i[0]+13, i[1]+13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2,
                    cv2.LINE_AA)

    dis(image_c)
