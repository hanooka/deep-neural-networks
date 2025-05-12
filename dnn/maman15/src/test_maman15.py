import torch
import pytest
import itertools

from .maman15 import myConv2d

RANDOM_STATE = 1337
# --- PARAM SPACE --- #
# one can add/remove params.
BATCH_SIZES = [1, 2]
IN_CHANNELS = [1, 2]
OUT_CHANNELS = [1, 2]
KERNEL_SIZES = [(3, 3), (5, 5)]
STRIDES = [1, 2]
PADDINGS = [0, 1]
INPUT_SHAPES = [(16, 16), (28, 28)]  # H x W


# We're testing the product of all params.
# 128 tests <=> 2^7
@pytest.mark.parametrize(
    "batch_sizes, in_channels, out_channels, kernel_size, stride, padding, input_shapes",
    itertools.product(BATCH_SIZES, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZES, STRIDES, PADDINGS, INPUT_SHAPES)
)
def test_custom_conv_matches_pytorch(
        batch_sizes, in_channels, out_channels, kernel_size, stride, padding, input_shapes
):
    H, W = input_shapes
    torch.manual_seed(RANDOM_STATE)

    # Input tensor
    x = torch.randn(batch_sizes, in_channels, H, W)

    # PyTorch Conv2d
    pytorch_conv2d = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True
    )

    # Our custom conv2d
    custom_conv2d = myConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # We're copying torch conv2d weights and biases to use with our own conv
    conv_pt_weight = pytorch_conv2d.weight.data.clone()
    conv_pt_bias = pytorch_conv2d.bias.data.clone()

    # overwrite our custom weights and biases with torch generated
    custom_conv2d.kernel.data = conv_pt_weight
    custom_conv2d.bias.data = conv_pt_bias

    # Forward pass
    out_pt = pytorch_conv2d(x)
    out_my = custom_conv2d(x)

    assert out_pt.shape == out_my.shape, f"Shape mismatch: {out_pt.shape} vs {out_my.shape}"

    # Check the tensors pretty much the same
    assert torch.allclose(out_my, out_pt, rtol=1e-4, atol=1e-5), \
        (f"Values differ for config: "
         f"in_channels={in_channels}, "
         f"out_channels={out_channels}, "
         f"kernel_size={kernel_size}, "
         f"stride={stride}, "
         f"padding={padding}")
