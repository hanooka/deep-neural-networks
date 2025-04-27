from collections import OrderedDict

import torch
from torch import nn


def init_weights(input_dim, output_dim):

class SplitLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        output_dim = input_dim
        assert input_dim % 2 == 0, f"input_dim: {input_dim} should be even."

        self.network = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(input_dim // 2, output_dim // 2)),
            ("a1", nn.ReLU())
        ]))

        print(self.network.sl)

    def forward(self, x: torch.Tensor):
        assert x.shape[1] % 2 == 0, f"x.shape[1]: {x.shape[1]} should be even."
        x1, x2 = x.split(x.shape[1] // 2, dim=-1)
        out1, out2 = self.network(x1), self.network(x2)
        return torch.cat([out1, out2], dim=-1)


if __name__ == '__main__':
    N = 2
    M = 4

    model = SplitLinear(M)
    print(model)
    x = torch.rand((N, M))

    print(x)
    y = model(x)
    print(y)

    print(x.shape)
    print(y.shape)

