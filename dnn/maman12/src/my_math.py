from __future__ import annotations

from collections import defaultdict

import torch
from typing import Union


class MyScalar:
    def __init__(self,
                 val: Union[int, float],
                 im_derivative: Union[int, float] = None,
                 predecessor: MyScalar = None):
        self.val = val
        self.im_derivative = im_derivative
        self.predecessor = predecessor

    # Nice repr/print for debugging.
    def __repr__(self):
        return (f"(MyScalar:\n"
                f"\tval: {self.val}\n"
                f"\tderivative: {self.im_derivative}\n"
                f"\tpredecessor:\n{self.predecessor})")


def power(x: MyScalar, var: Union[int, float]) -> MyScalar:
    def pow_der():
        if var == 0:
            return x.val
        # power derivative
        return torch.pow(var * x.val, torch.tensor(var - 1)).item()

    # calculating MyScalar 3 values...
    val = torch.pow(x.val, torch.tensor(var)).item()
    im_derivative = pow_der()

    return MyScalar(val, im_derivative, x)


def exp(x: MyScalar) -> MyScalar:
    # e^a == der(e^a), therefor, easy calculation
    val = torch.exp(torch.tensor(x.val)).item()
    return MyScalar(val, val, x)


def log(x: MyScalar) -> MyScalar:
    # the natural log (log_base_e)
    val = torch.log(torch.tensor(x.val)).item()
    im_derivative = 1. / x.val
    return MyScalar(val, im_derivative, x)


def sin(x: MyScalar):
    val = torch.sin(torch.tensor(x)).item()
    im_derivative = torch.cos(torch.tensor(x)).item()
    return MyScalar(val, im_derivative, x)


def cos(x: MyScalar):
    val = torch.cos(torch.tensor(x)).item()
    im_derivative = -torch.sin(torch.tensor(x)).item()
    return MyScalar(val, im_derivative, x)


def mult(x: MyScalar, var: Union[int, float]):
    val = x.val * var
    return MyScalar(val, var, x)


def add(x: MyScalar, var: Union[int, float]):
    val = x.val + var
    return (val, 1, x)


def get_gradient(x: MyScalar) -> dict:
    if not x:
        return {}

    result = {0: 1}

    # applying chain rule by cum_mult the derivatives
    i = 1
    while x.predecessor:
        result[i] = result[i - 1] * x.im_derivative
        i += 1
        x = x.predecessor

    # Renaming ints to chrs, and reverse the order of keys.
    # If more than 26 variables required, we can do modulo 26, and start enumerating the variables...
    result = {chr(ord('a') + key): val for key, val in zip(reversed(result.keys()), result.values())}
    return result


# TODO: Beautify and add TESTS.

if __name__ == '__main__':
    a = MyScalar(2)
    b = power(a, 2)
    c = exp(b)
    d = power(c, 2)
    print(d)
    e = get_gradient(d)
    print(e)
