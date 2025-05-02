import math
from collections import OrderedDict

import torch
from torch import nn
from torch.backends.mkl import verbose


def init_weights_pre_relu(input_dim, output_dim):
    """ Since we're using RELU activation, we'll implement the `he` initialization.
    I have ignored bias initialization problems, as we've got no "real training".
    No consideration on imbalance etc.
    We can test this using statistics and run few simulations to approx results with expectancy
    """
    std = math.sqrt(2 / input_dim)
    weights = torch.randn((input_dim, output_dim)) * std
    return weights


class SplitLinear(nn.Module):
    def __init__(self, input_dim, verbose=False):
        super().__init__()
        self.verbose = verbose
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

    def set_verbose(self, verbose):
        self.verbose = verbose

    def forward(self, x: torch.Tensor):
        assert x.shape[1] % 2 == 0, f"x.shape[1]: {x.shape[1]} should be even."
        x1, x2 = x.split(x.shape[1] // 2, dim=-1)
        if self.verbose:
            print(f"x1: {x1}\nx2: {x2}")
        out1, out2 = self.network(x1), self.network(x2)
        if self.verbose:
            print(f"out1: {out1}\nout2: {out2}")
        return torch.cat([out1, out2], dim=-1)


def q1():
    N = 2  # Batch size
    M = 4  # Features (1d)

    model = SplitLinear(M, verbose=True)
    x = torch.rand((N, M))

    print(x)
    y = model(x)
    print(y)
    print(x.shape)
    print(y.shape)
    print(f"Shapes equal: {x.shape == y.shape}")


nn.Dropout


class DropNorm(nn.Module):
    def __init__(self, drop_p=0.5):
        super().__init__()
        assert 0. <= drop_p <= 1., f"drop probability {drop_p} should be in between [0, 1]"
        self.drop_p = drop_p
        self.eps = 1e-16

    def dropout(self, x: torch.Tensor):
        # Case not training or p = 0
        if not self.training or self.drop_p == 0.:
            return x

        # Separating the batch from features ==> mask is shared across the whole batch.
        feature_shape = x.shape[1:]
        # Creating the mask using x.shape for randoms, and drop_p for 0/1
        mask = (torch.rand(feature_shape, device=x.device) > self.drop_p).float()
        # division: since we "activate" fewer neurons, we give them "more power"
        mask = mask / (1. - self.drop_p)
        # element wise multiplication ==> "masking"
        return x * mask

    def normalize(self, x):
        if not self.training:
            return x

        # We want all dims EXCEPT the batch dim, to be included in the mean
        # meaning every batch will have it's own mew, sig2, and eventually norm_x.
        dims = tuple(range(1, x.dim()))
        mew = torch.mean(x, dtype=torch.float32, dim=dims, keepdim=True)
        sig2 = torch.sum((x - mew) ** 2, dim=dims, keepdim=True) / math.prod(x.shape[1:])
        norm_x = (x - mew) / torch.sqrt(sig2 + self.eps)
        return norm_x

    def forward(self, x):
        # TODO: implement gama and betas for the network
        out1 = self.dropout(x)
        out2 = self.normalize(out1)
        return out2


def norm_example():
    x = torch.arange(0, 3 * 10 * 10).reshape(3, 10, 10)
    print(x)

    # We want all dims EXCEPT the batch dim, to be included in the mean
    dims = tuple(range(1, x.dim()))
    mew = torch.mean(x, dtype=torch.float32, dim=dims, keepdim=True)
    sig2 = torch.sum((x - mew) ** 2, dim=dims, keepdim=True) / math.prod(x.shape[1:])
    eps = 1e-16

    norm_x = (x - mew) / torch.sqrt(sig2 + eps)
    print(norm_x)

def main():
    pass


if __name__ == '__main__':
    main()
