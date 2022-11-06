import torch


def grad_acc():
    weights = torch.ones(4, requires_grad=True)
    for epoch in range(3):
        model_output = (weights * 3).sum()
        model_output.backward()
        print(weights.grad)
        # RESET GRADIENT (OTHERWISE IT ACCUMULATES)
        weights.grad.zero_()


def optimizer_example():
    weights = torch.ones(4, requires_grad=True)
    optimizer = torch.optim.SGD(weights, lr=0.01)
    optimizer.step()
    optimizer.zero_grad()


def intro():
    x = torch.randn(3, requires_grad=True)
    print(x)

    y = x + 2
    print(y)
    z = y * y * 2
    # z = z.mean()
    print(z)

    v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
    z.backward(v)  # gradient of z with respect to x: dx/dx
    print(x.grad)

    #   x.requires_grad_(False)
    #   x.detach()
    #   with torch.no_grad():

    with torch.no_grad():
        y = x + 2
        print(y)


if __name__ == '__main__':
    # intro()
    grad_acc()
