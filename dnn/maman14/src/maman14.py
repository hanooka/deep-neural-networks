import math
from collections import OrderedDict

import torch
from torch import nn


def init_weights_pre_relu(input_dim, output_dim):
    """ Since we're using RELU activation, we'll implement the `he` initialization. """
    std = math.sqrt(2/input_dim)
    weights = torch.randn((input_dim, output_dim)) * std
    return weights

class SplitLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        output_dim = input_dim
        assert input_dim % 2 == 0, f"input_dim: {input_dim} should be even."

        self.network = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(input_dim // 2, output_dim // 2)),
            ("a1", nn.ReLU())
        ]))
        # Custom weights creation!
        he_weights = init_weights_pre_relu(input_dim // 2, output_dim // 2)
        he_weights.requires_grad = True
        custom_weight = nn.Parameter(he_weights)
        self.network.l1.weight = custom_weight
        print(self.network.l1.weight)

    def forward(self, x: torch.Tensor):
        assert x.shape[1] % 2 == 0, f"x.shape[1]: {x.shape[1]} should be even."
        x1, x2 = x.split(x.shape[1] // 2, dim=-1)
        out1, out2 = self.network(x1), self.network(x2)
        return torch.cat([out1, out2], dim=-1)


if __name__ == '__main__':
    N = 2
    M = 4

    scaled = init_weights_pre_relu(M//2, M//2)
    print(scaled)

    model = SplitLinear(M)
    print(model)
    x = torch.rand((N, M))

    print(x)
    y = model(x)
    print(y)

    print(x.shape)
    print(y.shape)

