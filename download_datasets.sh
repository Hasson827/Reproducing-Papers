#!/bin/bash

# 创建 Datasets 文件夹（如果不存在）
mkdir -p Datasets
cd Datasets

# 下载 MNIST 数据集
echo "🔄 正在下载 MNIST 数据集..."
if [ ! -d "MNIST" ]; then
    mkdir MNIST
    cd MNIST
    wget --continue http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    wget --continue http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    wget --continue http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    wget --continue http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    echo "🗜️ 解压 MNIST 文件..."
    gunzip *.gz
    cd ..
else
    echo "✅ MNIST 数据集已存在，跳过下载。"
fi

# 下载 tiny-imagenet-200 数据集
echo "🔄 正在下载 tiny-imagenet-200 数据集..."
if [ ! -d "tiny-imagenet-200" ]; then
    wget --continue http://cs231n.stanford.edu/tiny-imagenet-200.zip
    echo "🗜️ 解压 tiny-imagenet-200.zip ..."
    unzip -q tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
else
    echo "✅ tiny-imagenet-200 数据集已存在，跳过下载。"
fi

echo "🎉 所有数据集已成功下载并准备好！"
