from typing import Iterable

import matplotlib.pyplot as plt
import torch


def my_sampler(size: int, dist: Iterable[float], requires_grad: bool = False):
    """ Creating a tensor of samples of size `size` using `dist` as distribution.
    work similarly to numpy random choice.

    :param size: tensor size. if size=5 will return tensor of size 5
    :param dist: distribution of sampling, for example
                 dist[1] is the chance to sample the number 1
                 all numbers in dist should be positive and sum to 1
    :param requires_grad:
    :return: torch.Tensor of dtype int32
    """
    dist = torch.Tensor(dist)

    is_dist_correct = sum(dist) == 1. and all(dist > 0.)
    if not is_dist_correct:
        raise ValueError(f"dist: {dist} is invalid.\n"
                         f"Hint: all values should be positive and sum to 1")

    # Once we cumsum the probability vector, we can find a random variable place in it.
    dist_cumsum = torch.cumsum(dist, dim=0, dtype=torch.float32)

    # Sampling `size` uniform values
    uni_randoms = torch.rand(size)

    # Searching in sorted array could be performed using binary search.
    # I can implement it, but why not using a torch implementation, which already
    # handles all the edge cases

    # We will also vectorize the operation instead running in a for loop
    result_tensor = torch.searchsorted(dist_cumsum, uni_randoms, out_int32=True)

    return result_tensor


def test1():
    dist = [0.7, 0.2, 0.1]
    result_tensor = my_sampler(20, dist)
    print(result_tensor)

def test2():
    dist = [0.7, 0.2, 0.1]
    result_tensor = my_sampler(10_000, dist)
    plt.hist(result_tensor, bins=[0, 1, 2, 3], linewidth=1.2)
    plt.show()


if __name__ == '__main__':
    test2()