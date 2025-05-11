import torch
from torch import nn


def pad(x: torch.Tensor, padding: int):
    b, c, h, w = x.shape
    x = torch.cat([
        torch.zeros(size=(b, c, padding, 1)),
        x,
        torch.zeros(size=(b, c, padding, 1))
    ], dim=2)
    b, c, h, w = x.shape
    x = torch.cat([
        torch.zeros(size=(b, c, 1, padding)),
        x,
        torch.zeros(size=(b, c, 1, padding))
    ], dim=3)
    return x


class myConv2d(nn.Module):
    """
    Padding - padding technique - add 0s
    Stride - single value for both dimensions (h, w)
    """

    def _check_input(self, in_channels, out_channels, kernel_size, stride, padding):
        assert in_channels > 0
        assert out_channels > 0
        assert len(kernel_size) == 2
        assert stride > 0
        assert padding >= 0

    def __init__(self, in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, padding=0):
        super().__init__()
        self._check_input(in_channels, out_channels, kernel_size, stride, padding)
        # kh = kernel height, kw = kernel width
        self.kh, self.kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        # Imagine (in_channels, *kernel_size) as a 3d cube, striding over the input.
        # Do that out_channels times to get the output
        self.kernel = nn.Parameter(torch.randn(
            (out_channels, in_channels, *kernel_size)
        ))
        self.bias = nn.Parameter(torch.randn((1, out_channels)))

    def forward(self, x: torch.Tensor):
        # We know the input should be 4 dims (Batch, C, H, W)
        assert len(x.shape) == 4, "Input shape: (B, C, H, W)"
        # holding batch_size, original_height, original_width
        # calculating output dims
        bs, oh, ow = x.shape[0], x.shape[2], x.shape[3]
        output_h = 1 + (oh - self.kh + self.padding * 2) // self.stride
        output_w = 1 + (ow - self.kw + self.padding * 2) // self.stride
        # Since we use for loops (task constraint) We will initialize memory for our output
        output = torch.empty(
            bs,
            self.out_channels,
            output_h,
            output_w
        )
        if self.padding > 0:
            x = pad(x, self.padding)
        for h in range(output_h):
            for w in range(output_w):
                # Because we stride we need to write hs (h_start) * stride, same for w
                hs = h * self.stride
                ws = w * self.stride
                sub_img = x[:, :, hs: hs + self.kh, ws: ws + self.kw]
                # tensordot, a beautiful operator that lets us do a dot product on wanted dims,
                # We've aligned the correct dims together so it looks nice here
                # the rest of the dims (B for x and out_channels (O) for self.kernel)
                # do a cartesian multiplication, which gives us the BxO result wanted, for each h and w
                # in our new output. Final output shape: (B, O, o_h, o_w)
                output[:, :, h, w] = torch.tensordot(sub_img, self.kernel, dims=([1, 2, 3], [1, 2, 3])) + self.bias
