#!/usr/bin/env python
# coding: utf-8

import os
import time
import numbers
import logging
import errno
import requests

from datetime import datetime

import torch
import torchvision.transforms as transforms

from cnn_finetune import make_model
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision.transforms.functional import center_crop, resize


def custom_seven_crop(img, size, interpolation):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    shift_w = int(round(w - crop_w) / 4.)
    shift_h = int(round(h - crop_h) / 4.)

    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    semi_full = resize(img.crop((shift_w, shift_h, w - shift_w, h - shift_h)), (crop_h, crop_w),
        interpolation=interpolation)
    full = resize(img, (crop_h, crop_w), interpolation=interpolation)
    return (tl, tr, bl, br, center, semi_full, full)


class CustomSevenCrop(object):
    def __init__(self, size, interpolation):
        self.size = size
        self.interpolation = interpolation
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_seven_crop(img, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.\
            format(self.size, self.interpolation)


class PytorchClassifier:
    def __init__(self, model_file, scale_size=224, scale_size_tta=256,
                 input_size=224, input_size_tta=224, topk=3, use_cuda=False):
        self.model_file = model_file
        self.scale_size = scale_size
        self.scale_size_tta = scale_size_tta
        self.input_size = input_size
        self.input_size_tta = input_size_tta
        self.topk = topk
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model, self.class_names, self.rgb_mean, self.rgb_std, self.interpolation = self.load_model(self.model_file)
        print("=> load model:", self.model_file)
        print("=> rgb_mean: {}  rgb_std: {}  interpolation: {}".format(self.rgb_mean, self.rgb_std, self.interpolation))
        print("=> scale_size: {}  input_size: {}".format(self.scale_size, self.input_size))
        print("=> scale_size_tta: {}  input_size_tta: {}".format(self.scale_size_tta, self.input_size_tta))
        print("=> topk:", self.topk)
        print("=> use CUDA:", self.use_cuda and torch.cuda.is_available())
        self.preprocess = self.get_preprocess(self.scale_size, self.input_size, self.rgb_mean, self.rgb_std, self.interpolation, False)
        self.preprocess_tta = self.get_preprocess(self.scale_size_tta, self.input_size_tta, self.rgb_mean, self.rgb_std, self.interpolation, True)
        self.softmax = torch.nn.Softmax(dim=1)

    def load_model(self, model_file):
        checkpoint = torch.load(model_file, map_location=self.device)
        args = checkpoint['args']

        try:
            rgb_mean = [float(mean) for mean in args.rgb_mean.split(',')]
            rgb_std = [float(std) for std in args.rgb_std.split(',')]
        except AttributeError:
            rgb_mean = args.rgb_mean
            rgb_std = args.rgb_std
        try:
            interpolation = args.interpolation
        except AttributeError:
            interpolation = 'BILINEAR'

        model_arch = checkpoint['arch']
        num_classes = checkpoint.get('num_classes', 0)
        if model_arch.startswith('efficientnet'):
            model = EfficientNet.from_name(model_arch)
            num_features = model._fc.in_features
            model._fc = torch.nn.Linear(num_features, num_classes)
        else:
            model = make_model(model_arch,
                               num_classes=num_classes,
                               pretrained=False)
        model.load_state_dict(checkpoint['model'])
        class_names = checkpoint.get('class_names', [])
        model.to(self.device)
        model.eval()
        return model, class_names, rgb_mean, rgb_std, interpolation

    def get_preprocess(self, scale_size, input_size, rgb_mean, rgb_std, interpolation, use_tta):
        interpolation = getattr(Image, interpolation, 2)

        if use_tta:
            preprocess = transforms.Compose([
                transforms.Resize(scale_size, interpolation=interpolation),
                CustomSevenCrop(input_size, interpolation),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(scale_size, interpolation=interpolation),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rgb_std)
            ])
        return preprocess

    # taken from https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/test_utils.py
    def download(self, url, fname=None, dirname=None, overwrite=False, retries=5):
        if fname is None:
            fname = url.split('/')[-1]

        if dirname is None:
            dirname = os.path.dirname(fname)
        else:
            fname = os.path.join(dirname, fname)
        if dirname != "":
            if not os.path.exists(dirname):
                try:
                    logging.info('create directory %s', dirname)
                    os.makedirs(dirname)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise OSError('failed to create ' + dirname)

        if not overwrite and os.path.exists(fname):
            logging.info("%s exists, skipping download", fname)
            return fname

        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                r = requests.get(url, stream=True)
                assert r.status_code == 200, "failed to open %s" % url
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                    break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, 's' if retries > 1 else ''))
        logging.info("downloaded %s into %s successfully", url, fname)
        return fname

    # taken from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    def pil_loader(self, filepath):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(filepath, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def get_image_file(self, filepath, use_tta=False):
        img_pil = self.pil_loader(filepath)
        if use_tta:
            img_tensor = self.preprocess_tta(img_pil)
        else:
            img_tensor = self.preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        return img_tensor.to(self.device)

    def get_image_url(self, url, use_tta=False):
        now = datetime.now()
        epoch = int(time.mktime(now.timetuple()))
        filename = "%d.jpg" % (epoch * 1000000 + now.microsecond)
        fname = self.download(url, fname=filename, dirname='test_images')
        return self.get_image_file(fname, use_tta=use_tta)

    def predict(self, img_variable, use_tta=False):
        with torch.no_grad():
            if use_tta:
                bs, ncrops, c, h, w = img_variable.size()
                output = self.model(img_variable.view(-1, c, h, w))
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output = self.model(img_variable)
        probs, labels = output.topk(self.topk)

        results = []
        for preds in self.softmax(output):
            probs, labels = preds.topk(self.topk)
            for probability, label in zip(probs, labels):
                results.append({"term": self.class_names[label], "score": float(probability)})
        return results

    def predict_url(self, url, use_tta=False):
        img_variable = self.get_image_url(url, use_tta=use_tta)
        return self.predict(img_variable, use_tta=use_tta)

    def predict_file(self, filepath, use_tta=False):
        img_variable = self.get_image_file(filepath, use_tta=use_tta)
        return self.predict(img_variable, use_tta=use_tta)
