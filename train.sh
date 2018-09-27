#!/bin/bash

loop_time=10
epoch=20
branch=16

for((i=1;i<=${loop_time};i++));  
do
echo ${i}"/"${loop_time}
python main.py train -ti ../Data/car_recognition/train -tl ../Data/car_recognition/train/labels_normal.txt -vi ../Data/car_recognition/test -vl ../Data/car_recognition/test/labels_normal.txt -b ${branch} -img-size 164 48 -label-len 7 -n ${epoch} -c checkpoints/weights_blue.h5 -pre checkpoints/weights_green.h5 -log log
python main.py train -ti ../Data/car_recognition/train -tl ../Data/car_recognition/train/labels_green.txt -vi ../Data/car_recognition/test -vl ../Data/car_recognition/test/labels_green.txt -b ${branch} -img-size 164 48 -label-len 8 -n ${epoch} -c checkpoints/weights_green.h5 -pre checkpoints/weights_blue.h5 -log log  
done