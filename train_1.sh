#!/bin/bash

python train.py train -ti ../Data/car_recognition/train -tl ../Data/car_recognition/train/labels_test.txt -vi ../Data/car_recognition/test -vl ../Data/car_recognition/test/labels_test.txt -b 8 -img-size 164 48 -label-len 7 -n 1 -c checkpoints/'test_weights.{epoch:02d}-{val_loss:.4f}.h5' -pre checkpoints/weights.h5 -log log

# python main.py train -ti ../Data/car_recognition/train -tl ../Data/car_recognition/train/labels_green.txt -vi ../Data/car_recognition/test -vl ../Data/car_recognition/test/labels_green.txt -b 16 -img-size 164 48 -label-len 8 -n 10 -c checkpoints/weights.h5 -pre checkpoints/weights.h5 -log log
