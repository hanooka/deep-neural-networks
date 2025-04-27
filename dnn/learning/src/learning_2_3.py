import torch
import torchvision

mnist_train = torchvision.datasets.FashionMNIST(
    root="/22961", train=True, download=True)
