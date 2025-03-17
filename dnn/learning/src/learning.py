import torch


def main():
    x = torch.arange(12, dtype=torch.float32)
    print(x)
    # number of elements
    x.numel()
    # reshapes into matrix of 3x4. notice 3x4 must = 12
    # also for automatic completion, 3, -1 or -1, 3 is possible.
    x.reshape(3, 4)
    # init a zeros tensor (OF SHAPE, 2, 3, 4)
    torch.zeros((2, 3, 4))
    # Same for ones
    torch.ones((2, 3, 4))
    # same with random numbers (gauss dist, 0, 1)
    torch.randn(3, 4)
    # outmost list is axis 0, inner is axis 1 and so on
    torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    # exp
    # note that operation exp : R ==> R
    torch.exp(x)

    # note that operations are syntactic sugar
    # developed by torch
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y, x - y, x * y, x / y, x ** y)

    # Broadcasting. We replicate a and b along their axis
    # to match each other. Then perform elementwise op.
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a, b)

    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))

    # This fucking trick with the [:] makes sure
    # we do not allocate new memory to Z. but overwrite
    # the previous memory location of Z.
    Z = torch.zeros_like(Y)
    print('id(Z):', id(Z))
    Z[:] = X + Y
    print('id(Z):', id(Z))

    # This holds true to operations like += /= -= etc

    # Converting back and forth from numpy/torch
    A = X.numpy()
    B = torch.from_numpy(A)
    type(A), type(B)

    # from tensor scalar to pythons native
    a = torch.tensor([3.5])
    a, a.item(), float(a), int(a)


    # torch.Tensor.expand_as()


if __name__ == '__main__':
    main()