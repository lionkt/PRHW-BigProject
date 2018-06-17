#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:19:01 2017

@author: Daniel Salo, Kevin Liang (Modifications)

Functions for testing Faster RCNN net and getting Mean Average Precision
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# from __future__ import print_function
import numpy as np
import os, sys, cv2
from tqdm import tqdm

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg


def calc_ap(rec, prec):
    """ ap = calc_ap(rec, prec, [use_07_metric])
    Compute AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_predictions(test_image_object, data_directory, names, ovthresh=0.5):
    """
    Evaluates predicted detections
    :param test_image_object: array, obj[cls][image] = N x 4 [x1, y1, x2, y2]
    :param data_directory: str, location of the evaluation folder.
    :param ovthresh: float, between 1 and 0, threshold for rejecting bbox
    :return: class_metrics: list of the APs of each class
    """
    # Get Ground Truth numbers for classes
    total_num = np.zeros([cfg.NCLASSES])

    # Labeled dict holds booleans for whether an object/gt_bbox has been counted yet
    labeled = {}

    print('Loading Ground Truth Data to count number of ground truth per class')
    for name in tqdm(names):  # number of test data
        name = name.split('.')[0]
        gt_boxes = np.loadtxt(data_directory + name + '.txt', ndmin=2, usecols=(2, 3, 4, 5))
        labeled[name] = []
        for g in range(gt_boxes.shape[0]):
            label = 1 #int(gt_boxes[g, 4])
            labeled[name].append(False)
            total_num[label] += 1
    print('Total Number of Objects per class: \n{0}'.format(total_num))

    # Define class_metrics list (skip background)
    class_metrics = np.zeros([cfg.NCLASSES - 1])

    # Calculate IoU for all classes and all images
    for c in range(1, cfg.NCLASSES):  # loop through all classes (skip background class)

        # Transform test_image_object into an np.array with all dets together.
        all_dets = test_image_object

        # Preallocate true positive and false positive arrays with zeros (number of detections)
        nd = all_dets.shape[0]
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # go down dets and mark TPs and FPs
        for d in range(nd):  # loop through all detections
            # Get ground truth
            img_indx = int(all_dets[d, -1])
            name = names[img_indx]
            name = name.split('.')[0]
            gt_boxes = np.loadtxt(data_directory + name + '.txt', ndmin=2, usecols=(2, 3, 4, 5))

            # Store proposal dets as bb and ground truth as bbgt
            bbgt = gt_boxes[:, :4]
            bb = all_dets[d, :4]

            # Compute Intersection Over Union
            ovmax, ovargmax = compute_iou(bbgt, bb)

            # Threshold
            if ovmax > ovthresh:
                if labeled[name][ovargmax] is False:  # ensure no ground truth box is double counted
                    tp[d] = 1
                    labeled[name][ovargmax] = True  # This is problematic.
                else:
                    fp[d] = 1
            else:
                fp[d] = 1

        # compute recall and precision
        cum_fp = np.cumsum(fp)
        cum_tp = np.cumsum(tp)
        rec = cum_tp / float(total_num[c])
        prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)  # avoid divide by zero

        # compute average precision and store
        ap = calc_ap(rec, prec)
        class_metrics[c - 1] = ap

    return class_metrics


def compute_iou(bbgt, bb):
    """ Computes the Intersection Over Union for a bounding box (bb) and a set of ground truth boxes (bbgt) """

    # compute intersection
    ixmin = np.maximum(bbgt[:, 0], bb[0])
    iymin = np.maximum(bbgt[:, 1], bb[1])
    ixmax = np.minimum(bbgt[:, 2], bb[2])
    iymax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # compute union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[:, 2] - bbgt[:, 0] + 1.) * (
                bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

    # computer IoU
    overlaps = inters / uni
    ovmax = np.max(overlaps)
    ovargmax = np.argmax(overlaps)

    return ovmax, ovargmax
