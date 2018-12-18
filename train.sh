#!/bin/bash

epoch=100
branch=24
lr=0.001
python train.py train -ti ../Data/car_recognition/train -tl ../Data/car_recognition/train/labels_normal.txt -vi ../Data/car_recognition/test -vl ../Data/car_recognition/test/labels_normal.txt -b ${branch} -img-size 164 48 -n ${epoch} -lr ${lr} -c checkpoints/weights_new.h5 -pre checkpoints/weights_new.h5 -log log

epoch=60
branch=24
lr=0.0001
python train.py train -ti ../Data/car_recognition/train -tl ../Data/car_recognition/train/labels_normal.txt -vi ../Data/car_recognition/test -vl ../Data/car_recognition/test/labels_normal.txt -b ${branch} -img-size 164 48 -n ${epoch} -lr ${lr} -c checkpoints/weights_new.h5 -pre checkpoints/weights_new.h5 -log log