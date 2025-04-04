import itertools

import numpy as np
import pytest
import torch
from .my_math import MyScalar, power, exp, log, sin, cos, mult, get_gradient


def assert_gradients_match(ms_grads: dict, pt_grads: dict, rel_tol=1e-5):
    assert set(ms_grads.keys()) == set(pt_grads.keys())
    for k in ms_grads:
        assert pytest.approx(ms_grads[k], rel=rel_tol) == pt_grads[k]


var_values = np.arange(-3.0, 4.5, 0.5)     # [1.0, 1.5, ..., 4.0]
var_values = var_values[var_values != 0]
exponent_values = np.arange(1.0, 4.0, 1.0) # [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "var, exponent",
    itertools.product(var_values, exponent_values)
)
def test_chain_exp_pow(var, exponent):
    """ Testing exp(pow(var));
    f(x) = exp(x^a) """

    # --- MyScalar ---
    a = MyScalar(var)
    b = power(a, exponent)
    c = exp(b)
    ms_grads = get_gradient(c)

    # --- PyTorch ---
    ta = torch.tensor(var, requires_grad=True)
    tb = ta ** exponent
    tb.retain_grad()
    tc = torch.exp(tb)
    tc.backward()

    pt_grads = {
        'c': 1.0,
        'b': tb.grad.item(),
        'a': ta.grad.item()
    }

    assert_gradients_match(ms_grads, pt_grads)


@pytest.mark.parametrize(
    "var", var_values
)
def test_chain_log_sin(var):
    # f(x) = sin(log(x))

    # --- MyScalar ---
    a = MyScalar(var)
    b = log(a)
    c = sin(b)
    ms_grads = get_gradient(c)  # { 'c': 1, 'b': ∂c/∂b, 'a': ∂c/∂a }

    # --- PyTorch ---
    ta = torch.tensor(var, requires_grad=True)
    tb = torch.log(ta)
    tb.retain_grad()
    tc = torch.sin(tb)
    tc.backward()

    pt_grads = {
        'c': 1.0,
        'b': tb.grad.item(),  # ∂c/∂b = cos(log(a))
        'a': ta.grad.item(),  # ∂c/∂a = ∂c/∂b * ∂b/∂a
    }

    assert_gradients_match(ms_grads, pt_grads)


def test_chain_mult_cos_exp_pow():
    # f(x) = 3 * cos(exp(x^2))

    a = MyScalar(2.0)
    b = power(a, 2)
    c = exp(b)
    d = cos(c)
    e = mult(d, 3)
    ms_grads = get_gradient(e)

    ta = torch.tensor(2.0, requires_grad=True)
    tb = ta ** 2
    tb.retain_grad()
    tc = torch.exp(tb)
    tc.retain_grad()
    td = torch.cos(tc)
    td.retain_grad()
    te = 3 * td
    te.backward()

    pt_grads = {
        'e': 1.0,
        'd': td.grad.item(),
        'c': tc.grad.item(),
        'b': tb.grad.item(),
        'a': ta.grad.item(),
    }

    assert_gradients_match(ms_grads, pt_grads)
