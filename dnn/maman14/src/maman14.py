import math
import torch

from torch import nn
from typing import Union
from collections import OrderedDict


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


class DropNormV0(nn.Module):
    def __init__(self, drop_p=0.5):
        super().__init__()
        assert 0. <= drop_p <= 1., f"drop probability {drop_p} should be in between [0, 1]"
        self.drop_p = drop_p
        self.eps = 1e-16

    def fail_dropout(self, x: torch.Tensor):
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
        return x * mask, mask

    def correct_dropout(self, x: torch.Tensor):
        # hard set of p to 0.5 like required.
        p = 0.5
        if not self.training:
            return x
        feature_shape = x.shape[1:]
        ele_num = math.prod(feature_shape)
        # bitwise check for `even` num
        assert ele_num & 1 == 0
        half_ele = ele_num // 2
        # Creating tensor with half 1 and half 0
        mask = torch.cat([torch.ones(half_ele, dtype=torch.float),
                          torch.zeroes(half_ele, dtype=torch.float)])
        # Generate random permutation (to order the 1s and 0s) <=> shuffle
        perm = torch.randperm(ele_num, device=x.device)
        # Shuffle the mask, reshape to original feature shape
        mask = mask[perm].reshape(feature_shape).to(x.device)
        return x * mask / p, mask

    def normalize(self, x):
        if not self.training:
            return x

        # We want all dims EXCEPT the batch dim, to be included in the mean
        # meaning every batch will have it's own mew, sig2, and eventually norm_x.
        dims = tuple(range(1, x.dim()))
        mew = torch.mean(x, dtype=torch.float32, dim=dims, keepdim=True)
        # std^2 | known also as `variance`
        sig2 = torch.sum((x - mew) ** 2, dim=dims, keepdim=True) / math.prod(x.shape[1:])
        norm_x = (x - mew) / torch.sqrt(sig2 + self.eps)
        return norm_x

    def dropout(self, x):
        return self.correct_dropout(x)

    def forward(self, x):
        # TODO: implement gama and betas for the network
        # TODO: Distinguish between training/infer
        out1 = self.dropout(x)
        out2 = self.normalize(out1)
        return out2


class DropNorm(nn.Module):
    def __init__(self, input_dim: Union[tuple, list, int]):
        super().__init__()
        self.eps = 1e-16
        # We init params so that y_i = x_i, similarly to batch norm
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def dropout(self, x: torch.Tensor):
        # hard set of p to 0.5 like required.
        p = 0.5
        if not self.training:
            return x
        feature_shape = x.shape[1:]
        ele_num = math.prod(feature_shape)
        # bitwise check for `even` num
        assert ele_num & 1 == 0
        half_ele = ele_num // 2
        # Creating tensor with half 1 and half 0
        mask = torch.cat([torch.ones(half_ele, dtype=torch.float, device=x.device),
                          torch.zeros(half_ele, dtype=torch.float, device=x.device)])
        # Generate random permutation (to order the 1s and 0s) <=> shuffle
        perm = torch.randperm(ele_num, device=x.device)
        # Shuffle the mask, reshape to original feature shape
        mask = mask[perm].reshape(feature_shape)
        return x * mask / p, mask

    def normalize(self, x):
        if not self.training:
            return x

        # We want all dims EXCEPT the batch dim, to be included in the mean
        # meaning every sample will have its own mew, sig2, and eventually norm_x.
        dims = tuple(range(1, x.dim()))
        mew = torch.mean(x, dtype=torch.float32, dim=dims, keepdim=True)
        # std^2 | known also as `variance`
        sig2 = torch.sum((x - mew) ** 2, dim=dims, keepdim=True) / math.prod(x.shape[1:])
        norm_x = (x - mew) / torch.sqrt(sig2 + self.eps)
        return norm_x

    def forward(self, x):
        """ When training, we use dropout -> normalization and we mult with mask as requested
            (we must multiply again with the mask, as beta might not be 0, and we want 0s)
        When not training, we only use normalize(x)*gamma + beta."""
        if self.training:
            out1, mask = self.dropout(x)
            out2 = self.normalize(out1)
            # We multiply at mask again because parameters that were zeroed in dropout should stay zeroed
            out2 = (self.gamma * out2 + self.beta) * mask
        else:
            out2 = self.gamma * self.normalize(x) + self.beta
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
