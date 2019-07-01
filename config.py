#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: plt.py
@time: 2019/6/20 10:03
@desc:
'''
import os

# base_dir = 'path to dataset base dir'
base_dir = './data'

image_dir = '/home/OCR/icdar2017/train'

image_dir1 = '/home/OCR/0325updated.task1train(626p)'

img_dir = os.path.join(base_dir, 'VOC2007/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007/Annotations')

train_txt_file = os.path.join(base_dir, r'VOC2007/ImageSets/Main/train.txt')
val_txt_file = os.path.join(base_dir, r'VOC2007/ImageSets/Main/val.txt')


anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]


checkpoints_dir = './checkpoints'
outputs = r'./logs'
