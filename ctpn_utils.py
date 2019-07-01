#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: plt.py
@time: 2019/6/20 10:21
@desc:
'''
import numpy as np
import cv2
from config import *
import datetime
import logging
import sys
import os

import numpy.random as npr


def get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')


def init_logger(log_file=None, log_path=None, log_level=logging.DEBUG, mode='w', stdout=True):
    """
    log_path: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_path is None:
        log_path = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.log'
    log_file = os.path.join(log_path, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(level=log_level,
                        format=fmt,
                        filename=os.path.abspath(log_file),
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width,height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def gen_anchor(featuresize, scale):
    """
        gen base anchor from feature map [HXW][9][4]
        reshape  [HXW][9][4] to [HXWX9][4]
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])
    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    # anchor = []
    # for i in shift_y:
    #     for j in shift_x:
    #         anchor.append(base_anchor + [j, i, j, i])
    # return np.array(anchor).reshape((-1, 4))
    len_x = shift_x.shape[0]
    len_y = shift_y.shape[0]

    x1 = np.asarray(list(shift_x)*len_y)
    y1 = np.asarray(list(shift_y)*len_x)
    y1 = y1.reshape([len_y,len_x]).T.reshape(-1)
    anchor = np.expand_dims(np.asarray([x1,y1,x1,y1]).T,1) + base_anchor
    return anchor.reshape([-1,4])


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    # for i in range(boxes1.shape[0]):
    #     overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    for i in range(boxes2.shape[0]):
        overlaps[:,i] = cal_iou(boxes2[i], area2[i], boxes1, area1)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def bbox_transfor_inv(anchor, regr):
    """
        return predict bbox
    """

    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5
    ha = anchor[:, 3] - anchor[:, 1] + 1

    Vcx = regr[0, :, 0]
    Vhx = regr[0, :, 1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    if gtboxes.shape[0] == 0:
        base_anchor = gen_anchor(featuresize, scale)

        labels = np.empty(base_anchor.shape[0])
        labels.fill(-1)
        bbox_targets = np.zeros([base_anchor.shape[0],2])
        bbox_inside_weights = np.zeros((base_anchor.shape[0], 2), dtype=np.float32)
        bbox_outside_weights = np.zeros((base_anchor.shape[0], 2), dtype=np.float32)

        return [labels,bbox_targets], base_anchor,[bbox_inside_weights,bbox_outside_weights]

    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_anchor(featuresize, scale)

#################################################################
    tf = False
    if tf:
        DEBUG = False
        _allowed_border = 0
        # only keep anchors inside the image
        # 仅保留那些还在图像内部的anchor，超出图像的都删掉
        inds_inside = np.where(
            (base_anchor[:, 0] >= -_allowed_border) &
            (base_anchor[:, 1] >= -_allowed_border) &
            (base_anchor[:, 2] < imgw + _allowed_border) &  # width
            (base_anchor[:, 3] < imgh + _allowed_border)  # height
        )[0]

        if DEBUG:
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = base_anchor[inds_inside, :]  # 保留那些在图像内的anchor
        if DEBUG:
            print('anchors.shape', anchors.shape)

        # 至此，anchor准备好了
        # --------------------------------------------------------------
        # label: 1 is positive, 0 is negative, -1 is dont care
        # (A)
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)  # 初始化label，均为-1

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt), shape is A x G
        # 计算anchor和gt-box的overlap，用来给anchor上标签
        # overlaps = bbox_overlaps(
        #     np.ascontiguousarray(anchors, dtype=np.float),
        #     np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
        overlaps = cal_overlaps(anchors, gtboxes)


        # 存放每一个anchor和每一个gtbox之间的overlap
        argmax_overlaps = overlaps.argmax(axis=1)  # (A)#找到和每一个gtbox，overlap最大的那个anchor
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)  # G#找到每个位置上9个anchor中与gtbox，overlap最大的那个
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_max_overlaps[np.where(gt_max_overlaps == 0)] = 1
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        EPS = 1e-14
        RPN_CLOBBER_POSITIVES = False
        RPN_NEGATIVE_OVERLAP = 0.3
        RPN_POSITIVE_OVERLAP = 0.7
        RPN_FG_FRACTION = 0.5
        RPN_BATCHSIZE = 300
        RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        RPN_POSITIVE_WEIGHT = -1.0

        RPN_PRE_NMS_TOP_N = 12000
        RPN_POST_NMS_TOP_N = 1000
        RPN_NMS_THRESH = 0.7
        RPN_MIN_SIZE = 8

        _counts = EPS
        _sums = np.zeros((1, 2))
        _squared_sums = np.zeros((1, 2))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

        if not RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0  # 先给背景上标签，小于0.3overlap的

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1  # 每个位置上的9个anchor中overlap最大的认为是前景
        # fg label: above threshold IOU
        labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1  # overlap大于0.7的认为是前景

        if RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        # 对正样本进行采样，如果正样本的数量太多的话
        # 限制正样本的数量不超过128个
        num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
            labels[disable_inds] = -1  # 变为-1

        # subsample negative labels if we have too many
        # 对负样本进行采样，如果负样本的数量太多的话
        # 正负样本总数是256，限制正样本数目最多128，
        # 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
        num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            # print "was %s inds, disabling %s, now %s inds" % (
            # len(bg_inds), len(disable_inds), np.sum(labels == 0))

        # 至此， 上好标签，开始计算rpn-box的真值
        # --------------------------------------------------------------
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = bbox_transfrom(anchors, gtboxes[argmax_overlaps, :])  # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(RPN_BBOX_INSIDE_WEIGHTS)  # 内部权重，前景就给1，其他是0

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if RPN_POSITIVE_WEIGHT < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0) + 1
            # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
            positive_weights = np.ones((1, 4))
            negative_weights = np.zeros((1, 4))
        else:
            assert ((RPN_POSITIVE_WEIGHT > 0) &
                    (RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (RPN_POSITIVE_WEIGHT /
                                (np.sum(labels == 1)) + 1)
            negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) /
                                (np.sum(labels == 0)) + 1)
        bbox_outside_weights[labels == 1, :] = positive_weights  # 外部权重，前景是1，背景是0
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            _sums += bbox_targets[labels == 1, :].sum(axis=0)
            _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            _counts += np.sum(labels == 1)
            means = _sums / _counts
            stds = np.sqrt(_squared_sums / _counts - means ** 2)
            print('means:')
            print(means)
            print('stdevs:')
            print(stds)

        def _unmap(data, count, inds, fill=0):
            """ Unmap a subset of item (data) back to the original set of items (of
            size count) """
            if len(data.shape) == 1:
                ret = np.empty((count.shape[0],), dtype=np.float32)
                ret.fill(fill)
                ret[inds] = data
            else:
                ret = np.empty((count.shape[0],) + data.shape[1:], dtype=np.float32)
                ret.fill(fill)
                ret[inds, :] = data
            return ret

        # map up to original set of anchors
        # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
        labels = _unmap(labels, base_anchor, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
        bbox_targets = _unmap(bbox_targets, base_anchor, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
        bbox_inside_weights = _unmap(bbox_inside_weights, base_anchor, inds_inside, fill=0)  # 内部权重以0填充
        bbox_outside_weights = _unmap(bbox_outside_weights, base_anchor, inds_inside, fill=0)  # 外部权重以0填充

        if DEBUG:
            print('rpn: max max_overlap', np.max(max_overlaps))
            print('rpn: num_positive', np.sum(labels == 1))
            print('rpn: num_negative', np.sum(labels == 0))
            _fg_sum += np.sum(labels == 1)
            _bg_sum += np.sum(labels == 0)
            _count += 1
            print('rpn: num_positive avg', _fg_sum / _count)
            print('rpn: num_negative avg', _bg_sum / _count)

        return [labels, bbox_targets], base_anchor ,[bbox_inside_weights,bbox_outside_weights]
    # labels
    # labels = labels.reshape((1, height, width, A))  # reshap一下label
    # rpn_labels = labels

    # bbox_targets
    # bbox_targets = bbox_targets \
    #     .reshape((1, height, width, A * 4))  # reshape

    # rpn_bbox_targets = bbox_targets
    # # bbox_inside_weights
    # bbox_inside_weights = bbox_inside_weights \
    #     .reshape((1, height, width, A * 4))
    #
    # rpn_bbox_inside_weights = bbox_inside_weights
    #
    # # bbox_outside_weights
    # bbox_outside_weights = bbox_outside_weights \
    #     .reshape((1, height, width, A * 4))
    # rpn_bbox_outside_weights = bbox_outside_weights

#################################################################
    else:
        # calculate iou
        overlaps = cal_overlaps(base_anchor, gtboxes)

        # init labels -1 don't care  0 is negative  1 is positive
        labels = np.empty(base_anchor.shape[0])
        labels.fill(-1)

        # for each GT box corresponds to an anchor which has highest IOU
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,range(overlaps.shape[1])]


        # the anchor with the highest IOU overlap with a GT box
        anchor_argmax_overlaps = overlaps.argmax(axis=1)
        anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

        # IOU > IOU_POSITIVE
        labels[anchor_max_overlaps > IOU_POSITIVE] = 1
        # IOU <IOU_NEGATIVE
        labels[anchor_max_overlaps < IOU_NEGATIVE] = 0
        # ensure that every GT box has at least one positive RPN region
        # labels[gt_max_overlaps > 0] = 1
        labels[gt_argmax_overlaps[gt_max_overlaps > 0]] = 1

        # only keep anchors inside the image
        outside_anchor = np.where(
            (base_anchor[:, 0] < 0) |
            (base_anchor[:, 1] < 0) |
            (base_anchor[:, 2] >= imgw) |
            (base_anchor[:, 3] >= imgh)
        )[0]
        labels[outside_anchor] = -1

        # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
        fg_index = np.where(labels == 1)[0]
        if (len(fg_index) > RPN_POSITIVE_NUM):
            labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

        # subsample negative labels
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            # print('bgindex:',len(bg_index),'num_bg',num_bg)
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

        # calculate bbox targets
        # debug here
        bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])
        # bbox_targets=[]

        bbox_inside_weights = np.zeros((base_anchor.shape[0], 2), dtype=np.float32)
        bbox_outside_weights = np.zeros((base_anchor.shape[0], 2), dtype = np.float32)

        return [labels, bbox_targets], base_anchor,[bbox_inside_weights,bbox_outside_weights]


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def shrink_poly(poly, r=16):
    # y = kx + b
    x_min = int(np.min(poly[:, 0]))
    x_max = int(np.max(poly[:, 0]))

    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]

    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    res = []

    start = int((x_min // 16 + 1) * 16)
    end = int((x_max // 16) * 16)

    p = x_min
    try:
        res.append([p, int(k1 * p + b1),
                    start - 1, int(k1 * (p + 15) + b1),
                    start - 1, int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])
    except BaseException as e:
        print(e)

    for p in range(start, end + 1, r):
        try:
            res.append([p, int(k1 * p + b1),
                    (p + 15), int(k1 * (p + 15) + b1),
                    (p + 15), int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])
        except BaseException as e:
            print(e)
    return np.array(res, dtype=np.int).reshape([-1, 8])


# for predict
class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    MAX_HORIZONTAL_GAP = 60
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)


class TextProposalConnectorOriented:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        # len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes

        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的全部小框
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs

