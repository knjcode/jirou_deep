#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import mxnet as mx
import cv2
import numpy as np
import heapq
from collections import namedtuple
import time
from datetime import datetime
import math

model_prefix = os.environ['MODEL_PREFIX']
model_epoch  = int(os.environ['MODEL_EPOCH'])
center_crop  = int(os.environ['CENTER_CROP'])
image_size   = int(os.environ['IMAGE_SIZE'])
rgb_mean     = [float(i) for i in os.environ['RGB_MEAN'].split(',')]
labels_txt   = os.environ['LABELS_TXT']

Batch = namedtuple('Batch', ['data'])
sym, arg_params, aux_params = mx.model.load_checkpoint('model/'+model_prefix, model_epoch)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False,
         data_shapes=[('data', (1, 3, image_size, image_size))])
mod.set_params(arg_params, aux_params, allow_missing=True)

labels = []
with open('model/'+labels_txt) as syn:
    labels = [l.split(' ')[-1].strip() for l in syn.readlines()]


def get_image(url):
    now = datetime.now()
    epoch = int(time.mktime(now.timetuple()))
    filename = "%d.jpg" % (epoch * 1000000 + now.microsecond)
    fname = mx.test_utils.download(url, fname=filename, dirname='test_images')
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if center_crop:
        img = crop_center(img)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32)
    img -= rgb_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def crop_center(img):
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    return img[yy:yy + short_edge, xx:xx + short_edge]


def create_blank(height, width, rgb_color):
    blank_img = np.zeros((height, width, 3), np.uint8)
    blank_img[:] = tuple(reversed(rgb_color))
    return blank_img


def padding_blank(image, left, top, right, bottom, color):
    height, width = image.shape[:2]
    pad_img = create_blank(height + top + bottom, width + left + right, color)
    pad_img[top:height + top, left:width + left] = image
    return pad_img


def resize_keep_aspect(image, target_width, target_height, color, interpolation=1):
    height, width = image.shape[:2]
    height_scale = float(target_height) / height
    width_scale = float(target_width) / width
    resize_scale = min(height_scale, width_scale)

    if (width >= height):
        roi_width = target_width
        roi_height = height * resize_scale
        roi_x = 0
        roi_y = int(math.floor((target_height - roi_height) / 2))
    else:
        roi_y = 0
        roi_height = target_height
        roi_width = width * resize_scale
        roi_x = int(math.floor((target_width - roi_width) / 2))

    roi_width = int(math.floor(roi_width))
    roi_height = int(math.floor(roi_height))

    resized_img = cv2.resize(image, (roi_width, roi_height), interpolation=interpolation)
    resized_img = padding_blank(resized_img, roi_x, roi_y, target_width - roi_width - roi_x, target_height - roi_height - roi_y, color)
    return resized_img


def predict(url):
    img = get_image(url)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    top3 = heapq.nlargest(3, enumerate(prob), key=lambda x: x[1])
    results = []
    for index, accuracy in top3:
        results.append({"term": labels[index], "score": float(accuracy)})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_url', help='specify image url')
    args = parser.parse_args()
    print(predict(args.image_url))
