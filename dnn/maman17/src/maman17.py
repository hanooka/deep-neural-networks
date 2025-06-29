import math
from unittest.mock import inplace

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        _input_dim = math.prod(input_dim)
        self.encoder = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(_input_dim, 256)),
            ("relu1", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(256, 128)),
            ("relu2", nn.ReLU(inplace=True)),
            ("out", nn.Linear(128, output_dim)),
        ]))

    def forward(self, x):
        out = self.encoder(x)
        return out


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, as_image=False):
        super().__init__()
        self.as_image = as_image
        self.output_dim = output_dim
        _output_dim = math.prod(output_dim)
        self.decoder = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_dim, 128)),
            ("relu1", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(128, 256)),
            ("relu2", nn.ReLU(inplace=True)),
            ("out", nn.Linear(256, _output_dim)),
            ("sig", nn.Sigmoid())
        ]))

    def forward(self, x):
        out = self.decoder(x)
        if self.as_image:
            out = out.view(-1, *self.output_dim)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoded_dim, as_image=False):
        super().__init__()
        self.input_dim = self.output_dim = input_dim
        self.encoded_dim = encoded_dim
        self.as_image = as_image
        self.encoder = Encoder(input_dim=self.input_dim, output_dim=encoded_dim)
        self.decoder = Decoder(input_dim=encoded_dim, output_dim=self.output_dim, as_image=as_image)

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out


def main():
    # Define transformation (convert to tensor and normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to PyTorch tensor (shape: C x H x W, value: [0,1])
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])

    # Download and load training dataset
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load test dataset
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Example of using the data loader
    for images, labels in train_loader:
        print(images.shape)  # e.g. torch.Size([64, 1, 28, 28])
        print(labels.shape)  # e.g. torch.Size([64])
        break  # just checking one batch


if __name__ == '__main__':
    main()
