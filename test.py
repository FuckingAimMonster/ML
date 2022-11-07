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
from core.config import cfg
from core.yolov3 import YOLOv3, decode


def center_crop(img):
    
    h, w, c = img.shape
    set_size = 700
    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = set_size

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def get_head_coord(img, bboxes):
    # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    x = int((bboxes[0][2] - bboxes[0][0]) / 2 + bboxes[0][0])
    y = int((bboxes[0][3] - bboxes[0][1]) * 1.5 / 14.57 + bboxes[0][1])
    print(x, y)
    cv2.line(img, (x, y), (x, y), (0,0,255), 3)
    return image


INPUT_SIZE   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

predicted_dir_path = 'C:/Users/yg058/Desktop/study/DeepLearning/capstone/TensorFlow2.0-Examples/4-Object_Detection/mAP/predicted'
ground_truth_dir_path = 'C:/Users/yg058/Desktop/study/DeepLearning/capstone/TensorFlow2.0-Examples/4-Object_Detection/mAP/ground-truth'
if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

# Build Model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):   
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)


cnt = 0
model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./yolov3")
model.trainable = False
with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        # image = center_crop(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt=[]
            classes_gt=[]
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
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
            image = get_head_coord(image,bboxes)
            # cv2.imshow("image", image)
            # cv2.waitKey() 
            cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH+image_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())

