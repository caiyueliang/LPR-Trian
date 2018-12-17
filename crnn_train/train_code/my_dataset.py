#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image, ImageFilter
import numpy as np
import cv2

# # 图片加载类
# class MyDataset(data.Dataset):
#     def __init__(self, root_dir, label_file, img_size, transforms=None, is_train=False):
#         self.root_dir = root_dir
#         records_txt = common.read_data(label_file, 'r')
#         self.records = records_txt.split('\n')
#         self.img_size = img_size
#         self.is_train = is_train
#
#         # imgs = os.listdir(root)
#         # self.imgs = [os.path.join(root, img) for img in imgs]
#         # self.label_path = label_path
#         self.transforms = transforms
#
#     def __getitem__(self, index):
#         record = self.records[index]
#         str_list = record.split(" ")
#         img_file = os.path.join(self.root_dir, str_list[0])
#
#         # print('img_file', img_file)
#         img = Image.open(img_file)
#
#         label = str_list[2:]
#         label = map(float, label)
#         label = np.array(label)
#
#         if self.is_train:                                               # 训练模式，才做变换
#             # img, label = self.RandomHorizontalFlip(img, label)        # 图片做随机水平翻转
#             img, label = self.random_crop(img, label)                   # 图片做随机裁剪
#             # self.show_img(img, label)
#
#         old_size = img.size[0]
#         label = label * self.img_size / old_size
#         if self.transforms:
#             img = self.transforms(img)
#
#         return img, label, img_file
#
#     def __len__(self):
#         return len(self.records)
#
#     # 图片做随机水平翻转
#     def RandomHorizontalFlip(self, img, label, p=0.5):
#         if random.random() < p:
#             w, h = img.size
#             img = functional.hflip(img)
#             for i in range(len(label)/2):
#                 label[2*i] = w - label[2*i]
#
#         return img, label
#
#     # 随机裁剪
#     def random_crop(self, img, labels):
#         # print('random_crop', labels)
#         # mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
#         # random.randrange(int(0.3*short_size), short_size)
#         img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#         # cv2.imshow('img_cv', img_cv)
#
#         imh, imw, _ = img_cv.shape
#         # short_size = min(imw, imh)
#         # print(imh, imw, short_size)
#
#         left = min(labels[0], labels[4])
#         top = min(labels[1], labels[3])
#         min_left = min(left, top)
#
#         right = min(imw-labels[2], imw-labels[6])
#         bottom = min(imh-labels[5], imh-labels[7])
#         min_right = min(right, bottom)
#
#         # print('left, top, right, bottom', left, top, right, bottom)
#         # print('min_left, min_right', min_left, min_right)
#
#         x1 = 0
#         y1 = 0
#         x2 = imw
#         y2 = imh
#
#         if random.random() < 0.5:
#             rate = random.random()
#             crop = int(min_left * rate)
#             x1 = crop
#             labels[0] = labels[0] - crop
#             labels[2] = labels[2] - crop
#             labels[4] = labels[4] - crop
#             labels[6] = labels[6] - crop
#
#             y1 = crop
#             labels[1] = labels[1] - crop
#             labels[3] = labels[3] - crop
#             labels[5] = labels[5] - crop
#             labels[7] = labels[7] - crop
#
#         if random.random() < 0.5:
#             rate = random.random()
#             crop = int(min_right * rate)
#             x2 = imw - crop
#             y2 = imh - crop
#
#         # print('x1, y1, x2, y2', x1, y1, x2, y2)
#         img_cv = img_cv[y1:y2, x1:x2]
#         # cv2.imshow('crop_img_cv', img_cv)
#         # cv2.waitKey(0)
#
#         img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
#         return img, labels


class MyDataset(Dataset):
    def __init__(self, root=None, label_file=None, transform=None, target_transform=None, is_train=False):
        self.root = root
        self.label_file = label_file
        self.sample_list = list()
        self.is_train = is_train

        with open(self.label_file, 'r') as f:
            self.sample_list = f.readlines()

        self.nSamples = len(self.sample_list)
        # print(self.sample_list)
        print('label_file: %s samples len: %d' % (self.label_file, self.nSamples))

        # self.env = lmdb.open(
        #     root,
        #     max_readers=1,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False)
        #
        # if not self.env:
        #     print('cannot creat lmdb from %s' % (root))
        #     sys.exit(0)
        #
        # with self.env.begin(write=False) as txn:
        #     nSamples = int(txn.get('num-samples'))
        #     self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < self.nSamples, 'index range error'
        record = self.sample_list[index].replace('\n', '').replace('\r', '').replace(' ', '')
        str_list = record.split(':')
        label = str_list[-1]
        image_path = str_list[0]

        img = Image.open(os.path.join(self.root, image_path))

        if self.is_train is True:
            img = self.random_gaussian(img)
            img = self.random_crop(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        # index += 1
        # with self.env.begin(write=False) as txn:
        #     img_key = 'image-%09d' % index
        #     imgbuf = txn.get(img_key)
        #
        #     buf = six.BytesIO()
        #     buf.write(imgbuf)
        #     buf.seek(0)
        #     try:
        #         img = Image.open(buf).convert('L')
        #     except IOError:
        #         print('Corrupted image for %d' % index)
        #         return self[index + 1]
        #
        #     if self.transform is not None:
        #         img = self.transform(img)
        #
        #     label_key = 'label-%09d' % index
        #     label = str(txn.get(label_key))
        #
        #     if self.target_transform is not None:
        #         label = self.target_transform(label)

        # print(img)
        # print(img.size())
        # print(label)
        return (img, label)

    # 随机高斯模糊
    def random_gaussian(self, img, max_n=2):
        # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("old", img_1)

        k = random.random()
        if k > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
            # img = cv2.GaussianBlur(img, ksize=(k, k), sigmaX=1.5)

        # img_2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("new", img_2)
        # cv2.waitKey()
        return img

    # 随机裁剪
    def random_crop(self, img, max_n=5):
        imw, imh = img.size
        # img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow("old", img_1)

        top = random.randint(0, max_n)
        bottom = random.randint(0, max_n)
        left = random.randint(0, max_n)
        right = random.randint(0, max_n)
        # print top, bottom, left, right

        # (left, upper, right, lower)
        box = (left, top, imw-right, imh-bottom)
        roi = img.crop(box)
        roi = roi.resize((imw, imh))

        # print(roi.size)
        roi = roi.resize((imw, imh))
        # print(roi.size)
        # img_2 = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR)
        # cv2.imshow("new", img_2)
        # cv2.waitKey()

        return roi


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
