import torch
import pytest
from sympy import capture

from .maman11 import my_broadcast, is_broadcastable, my_broadcast_tensors


class TestMyBroadcast():
    @pytest.mark.parametrize(
        "a, b, expected_shape",
        [
            (torch.arange(3),                torch.randn((2, 3, 3)),         (2, 3, 3)),
            (torch.arange(2).reshape(2, 1),  torch.randn((2, 3)),            (2, 3)),
            (torch.arange(9).reshape(3, 3),  torch.randn((3, 3)),            (3, 3)),
            (torch.rand((3, 1)),             torch.randn((3, 5)),            (3, 5)),
        ]
    )
    def test_valid_broadcasts(self, a, b, expected_shape):
        c = my_broadcast(a, b)
        assert c.shape == expected_shape, f"Expected: {expected_shape}. Got: {c.shape}"

    @pytest.mark.parametrize(
        "a, b",
        [
            (torch.randn((2, 5)),       torch.randn((2, 3))),
            (torch.randn((1, 1, 2)),    torch.randn((2, 1, 1))),
            (torch.randn((2, 3)),       torch.randn((2, 3, 4)))
        ]
    )

    def test_invalid_broadcast(self, a, b):
        with pytest.raises(ValueError):
            my_broadcast(a, b)


class TestIsBroadcastable:
    @pytest.mark.parametrize(
        "a, b, expected_result",
        [
            (torch.arange(3),                torch.randn((2, 3, 3)),         (True, [2, 3, 3])),
            (torch.arange(2).reshape(2, 1),  torch.randn((2, 3)),            (True, [2, 3])),
            (torch.arange(9).reshape(3, 3),  torch.randn((3, 3)),            (True, [3, 3])),
            (torch.rand((3, 1)),             torch.randn((3, 5)),            (True, [3, 5])),
            (torch.randn((2, 5)),            torch.randn((2, 3)),            (False, None)),
            (torch.randn((2, 3)),            torch.randn((2, 3, 4)),         (False, None))
        ]
    )
    def test_is_broadcastable(self, a, b, expected_result):
        result = is_broadcastable(a, b)
        assert result == expected_result, f"Expected: {expected_result}. Got: {result}"


class TestMyBroadcastTensors():
    @pytest.mark.parametrize(
        "a, b",
        [
            (torch.randn((1, 1, 2)),        torch.randn((2, 1, 1))),
            (torch.arange(2).reshape(2, 1), torch.randn((2, 3))),
            (torch.arange(9).reshape(3, 3), torch.randn((3, 3))),
            (torch.rand((3, 1)),            torch.randn((3, 5))),
            (torch.rand((3, 1)),            torch.randn((3, 2, 3, 2, 1, 5))),
        ]
    )
    def test_broadcast_equality(self, a, b):
        # my custom broadcast
        x1, x2 = my_broadcast_tensors(a, b)

        # pytorch broadcast
        y1, y2 = torch.broadcast_tensors(a, b)

        assert torch.equal(x1, y1), f"Tensor 1 mismatch. Expected: {y1.shape}, Got: {x1.shape}"
        assert torch.equal(x2, y2), f"Tensor 2 mismatch. Expected: {y2.shape}, Got: {x2.shape}"

    @pytest.mark.parametrize(
        "a, b",
        [
            (torch.randn((2, 5)),       torch.randn((2, 3))),
            (torch.randn((1, 1, 2)),    torch.randn((2, 1, 1, 3))),
            (torch.randn((2, 3)),       torch.randn((2, 3, 4))),
            (torch.randn((3, 2, 1)),    torch.randn((2, 1, 3))),
        ]
    )
    def test_invalid_broadcast(self, a, b):
        # Test that RuntimeError is raised when broadcasting fails
        with pytest.raises(RuntimeError):
            my_broadcast_tensors(a, b)
