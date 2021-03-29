#!/bin/bash
# Clone darknet into current folder and build the project
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
# Download configuration file
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
# Download weight file
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# Run below for testing purpose: 
# ./darknet detector test cfg/coco.data yolov4.cfg yolov4.weights -ext_output data/dog.jpg
# Alternatively run the following: 
# python3 yolostage.py

# Download datasets
cd ../
# ./download.sh

# Install additional python dependencies
pip install matplotlib
