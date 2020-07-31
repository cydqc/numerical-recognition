#!/bin/sh

wget -nc http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz; gunzip -f train-images-idx3-ubyte.gz
wget -nc http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz; gunzip -f train-labels-idx1-ubyte.gz
wget -nc http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz; gunzip -f t10k-images-idx3-ubyte.gz
wget -nc http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz; gunzip -f t10k-labels-idx1-ubyte.gz
