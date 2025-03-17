from typing import Union, Optional, Tuple

import torch
from collections import deque

from sympy import capture


def broadcast_from_shape(a: torch.tensor, b_shape: Union[list | tuple]):
    """ Helper function for my_broadcast. Doing everything using the shape of b.
    This function created to fulfil the 3rd question of the maman.
    """

    # Cloning a. We don't want to override it (questions constraints)
    a = a.clone()
    # we use deque because we might append 1s from the left
    a_shape = deque(a.shape)

    # Match shapes, appending 1s to a shape while a is smaller
    while len(a_shape) < len(b_shape):
        a_shape.appendleft(1)

    # Validity of tensors dimensions check
    for dim_a, dim_b in zip(a_shape, b_shape):
        if dim_a != dim_b and dim_a != 1:
            raise ValueError(f"{a.shape} cannot be broad-casted to {b_shape}.")

    # reshape a to the correct basic shape
    a = a.reshape(tuple(a_shape))

    # Expand a along the dimensions to match b
    for i, (dim_a, dim_b) in enumerate(zip(a_shape, b_shape)):
        if dim_a == 1 and dim_b > dim_a:
            a = torch.cat([a] * dim_b, dim=i)

    return a


def my_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Give a, b, torch Tensors, we will reshape a as b by cloning a as much as needed.
    Return a new tensor which is shaped like b.
    Doesn't override a nor b.
    """
    b_shape = b.shape
    return broadcast_from_shape(a, b_shape)


def is_broadcastable(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, Optional[Union[list | tuple]]]:
    """ Given a, b, torch Tensors, check if it's possible to broadcast together.
    Returns Tuple:
        First element either True/False if it's possible.
        If True:
            Second Element is the new shape both tensors will be broad casted into
        If False:
            Second Element is None.
    """
    new_shape = []
    # Making sure a has fewer dimensions.
    if len(b.shape) < len(a.shape):
        a, b = b, a
    a_shape = deque(a.shape)
    b_shape = b.shape
    # equalizing dimensions.
    while len(a_shape) < len(b_shape):
        a_shape.appendleft(1)

    # If dimensions are different, and none of them is 1, it's impossible to broadcast.
    # Anything else is possible, and we want the bigger (either "bigger" or 1) dimension as the new dimension
    for dim_a, dim_b in zip(a_shape, b_shape):
        if dim_a != dim_b:
            if dim_a != 1 and dim_b != 1:
                return (False, None)

        new_shape.append(max(dim_a, dim_b))

    return (True, new_shape)


def my_broadcast_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    is_bc, bc_shape = is_broadcastable(a, b)
    if not is_bc:
        raise RuntimeError(f"{a.shape} and {b.shape} cannot be broad-casted together.")
    a = broadcast_from_shape(a, bc_shape)
    b = broadcast_from_shape(b, bc_shape)
    return a, b


if __name__ == '__main__':
    pass