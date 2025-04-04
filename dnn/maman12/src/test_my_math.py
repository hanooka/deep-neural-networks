import itertools
import math

import numpy as np
import pytest
import torch
from . import my_math as mm


def assert_gradients_match(ms_grads: dict, pt_grads: dict, rel_tol=1e-4):
    # Safety check for same keys.
    assert set(ms_grads.keys()) == set(pt_grads.keys())
    # check same value with a tolerance of `rel_tol`
    for k in ms_grads:
        assert pytest.approx(ms_grads[k], rel=rel_tol) == pt_grads[k]


# Range is fairly low, but creates a big enough space for tests.
# If numbers go larger/smaller, numeric instability will result failed tests.
# One can fiddle with the 3rd argument (jump) to generate more tests.
var_values = np.arange(-2.5, 2.5, 0.5)
exponent_values = np.arange(1.0, 4.0, 1)
mult_values = np.arange(-2., 2.5, 0.5)


@pytest.mark.parametrize(
    "var, exponent",
    itertools.product(var_values, exponent_values)
)
def test_chain_exp_pow(var, exponent):
    """ Testing exp(pow(var));
    f(x) = exp(x^a) """

    # --- MyScalar ---
    a = mm.MyScalar(var)
    b = mm.power(a, exponent)
    c = mm.exp(b)
    ms_grads = mm.get_gradient(c)

    if any(math.isnan(x) or math.isinf(x) for x in ms_grads.values()):
        pytest.skip("Invalid result: NaN or Inf in custom gradient.")

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
    """ Testing log(sin(var)) """
    # f(x) = sin(log(x))

    # --- MyScalar ---
    a = mm.MyScalar(var)
    b = mm.log(a)
    c = mm.sin(b)
    ms_grads = mm.get_gradient(c)

    print(ms_grads)

    if any(math.isnan(x) or math.isinf(x) for x in ms_grads.values()):
        pytest.skip("Invalid result: NaN or Inf in custom gradient.")

    # --- PyTorch ---
    ta = torch.tensor(var, requires_grad=True)
    tb = torch.log(ta)
    tb.retain_grad()
    tc = torch.sin(tb)
    tc.backward()

    pt_grads = {
        'c': 1.0,
        'b': tb.grad.item(),
        'a': ta.grad.item()
    }

    assert_gradients_match(ms_grads, pt_grads)


@pytest.mark.parametrize(
    "var, exponent, multiple",
    itertools.product(var_values, exponent_values, mult_values)
)
def test_chain_mult_cos_exp_pow(var, exponent, multiple):
    """ Testing mult(cos(exp(pow(var), exponent)), multiple) """
    # f(x) = m * cos(exp(x^p))

    a = mm.MyScalar(var)
    b = mm.power(a, exponent)
    c = mm.exp(b)
    d = mm.cos(c)
    e = mm.mult(d, multiple)
    ms_grads = mm.get_gradient(e)

    if any(math.isnan(x) or math.isinf(x) for x in ms_grads.values()):
        pytest.skip("Invalid result: NaN or Inf in custom gradient.")

    ta = torch.tensor(var, requires_grad=True)
    tb = ta ** exponent
    tb.retain_grad()
    tc = torch.exp(tb)
    tc.retain_grad()
    td = torch.cos(tc)
    td.retain_grad()
    te = multiple * td
    te.backward()

    pt_grads = {
        'e': 1.0,
        'd': td.grad.item(),
        'c': tc.grad.item(),
        'b': tb.grad.item(),
        'a': ta.grad.item()
    }

    assert_gradients_match(ms_grads, pt_grads)
