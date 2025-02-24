# Reproducing-Papers-Of-CNN

Welcome to the **Reproducing Classical Papers of CNN** repository. This repository is dedicated to reproducing the code for classical models from seminal papers in the field of Convolutional Neural Networks (CNNs).

## Table of Contents

- [Introduction](#introduction)
- [Papers and Models](#papers-and-models)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this repository is to provide implementations of classical CNN models as described in influential research papers. By reproducing these models, we aim to facilitate understanding and further research in the field of deep learning.

## Papers and Models

Here is a list of the papers and corresponding models that we have implemented:

1. **LeNet-5** - [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Yann LeCun et al.
2. **AlexNet** - [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) by Alex Krizhevsky et al.
3. **VGGNet** - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) by Karen Simonyan and Andrew Zisserman.
4. **GoogLeNet** - [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf) by Christian Szegedy et al.
5. **ResNet** - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He et al.
6. **U-Net** - [U-Net: Convolutional Networks for BiomedicalImage Segmentation](https://arxiv.org/pdf/1505.04597) by Olaf Ronneberger et al.

## Dataset Download Instructions

This project uses the following datasets:

* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [tiny-imagenet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

To download and prepare these datasets, please follow the steps below:

### 💾 Download Datasets

1. Make sure you have `wget` and `unzip` installed. You can install them on Linux/macOS using the following commands:

   ```bash
   sudo apt update && sudo apt install wget unzip   # Ubuntu/Debian
   brew install wget unzip                          # macOS (using Homebrew)
   ```
   
2.  Running the following command to download datasets:

    ```bash
    bash download_datasets.sh
    ```

## Contributing

We welcome contributions from the community. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
