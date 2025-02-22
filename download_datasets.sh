#!/bin/bash

# 创建 Datasets 文件夹（如果不存在）
mkdir -p Datasets
cd Datasets

# ✅ 下载 MNIST 数据集 (使用 pytorch 自动下载)
echo "🔄 正在下载 MNIST 数据集..."
if [ ! -d "MNIST" ]; then
    mkdir MNIST
    cd MNIST
    echo "📦 使用 Python 自动下载 MNIST 数据集..."
    /usr/bin/python3 -c "
import torchvision.datasets as datasets
datasets.MNIST(root='.', download=True)
"
    echo "✅ MNIST 数据集下载完成！"
    cd ..
else
    echo "✅ MNIST 数据集已存在，跳过下载。"
fi

# ✅ 下载 tiny-imagenet-200 数据集
echo "🔄 正在下载 tiny-imagenet-200 数据集..."
if [ ! -d "tiny-imagenet-200" ]; then
    wget --continue http://cs231n.stanford.edu/tiny-imagenet-200.zip
    echo "🗜️ 解压 tiny-imagenet-200.zip ..."
    unzip -q tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
    echo "✅ 解压完成，正在重新组织 val 文件夹..."

    # 🔄 重新组织 val 文件夹
    cd tiny-imagenet-200/val
    mkdir images_split
    awk '{print $1}' val_annotations.txt | sort | uniq | while read class; do
        mkdir -p images_split/$class
    done
    while IFS=$'\t' read -r file class _; do
        mv images/$file images_split/$class/
    done < val_annotations.txt
    rm -r images
    mv images_split images
    echo "✅ val 文件夹已成功重新组织！"
    cd ../..

else
    echo "✅ tiny-imagenet-200 数据集已存在，跳过下载。"
fi

echo "🎉 所有数据集已成功下载并准备好！"
