import torch
import numpy as np

if __name__ == '__main__':
    """" EMPTY TENSORS """
    print(f"1D tensor with 1 element: {torch.empty(1)}")
    print(f"1D tensor with 3 elements: {torch.empty(3)}")

    print(f"2D tensor with 6 elements:\n {torch.empty(2, 3)}")
    print(f"3D tensor with 12 elements:\n {torch.empty(2, 2, 3)}")

    print(f"Tensor with random elements:\n {torch.rand(2, 2)}")
    print(f"Tensor with 0 elements:\n {torch.zeros(2, 2)}")

    x = torch.ones(2, 2, dtype=torch.int)
    print(f"Tensor with 1 elements:\n {x}")
    print(f"Type of tensor: {x.dtype}")
    print(f"Size of tensor: {x.size()}")

    x = torch.tensor([2.5, 0.1])
    print(x)

    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    z = x + y
    print(z)
    print(y.add_(x))
    print(y)

    """" SLICING """
    x = torch.rand(5, 3)
    print(x)
    print(x[:, 0])
    print(x[1, :])
    print(x[1, 1])
    print(x[1, 1].item())

    """" RESHAPING """
    x = torch.rand(4, 4)
    print(x)
    y = x.view(16)
    print(y)
    y = x.view(-1, 8)
    print(y)

    """" TENSOR TO NUMPY """
    a = torch.ones(5)
    print(a)
    b = a.numpy()
    print(b)
    print(type(b))

    a.add_(1)
    print(a)
    print(b)

    """" NUMPY TO TENSOR """
    a = np.ones(5)
    print(a)
    b = torch.from_numpy(a)
    print(b)

    a += 1
    print(a)
    print(b)

    """" TENSOR ON GPU """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.ones(5, device=device)
        y = torch.ones(5)
        y = y.to(device)
        z = x + y
        print(z)
        z = z.to("cpu")
        print(z)

    """" REQUIRES GRAD (tell torch to calculate gradient for tensor, used if you want to optimize this tensor)"""
    x = torch.ones(5, requires_grad=True)
    print(x)
