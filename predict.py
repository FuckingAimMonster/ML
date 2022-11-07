#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
import test
from core.config import cfg
from core.yolov3 import YOLOv3, decode


def predict(imgs):

    INPUT_SIZE   = 416
    NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

    # Build Model
    input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    ret_bboxes = []
    for i, fm in enumerate(feature_maps):   
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights("./yolov3")
    model.trainable = False
    for image in enumerate(imgs):
        image = test.center_crop(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Predict Process
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
        np_bboxes = np.array(bboxes)
        print(np_bboxes.shape)
        if np_bboxes.shape[0] != 1:
            print("pass!")
            continue
        if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            image = utils.draw_bbox(image, bboxes)
            image = test.get_head_coord(image,bboxes)
            ret_bboxes.append(bboxes)
            # cv2.imshow("image", image)
            # cv2.waitKey()
    return bboxes
        

