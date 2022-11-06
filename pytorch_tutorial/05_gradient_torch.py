# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights
import torch
import torch.nn as nn


def torch_gradient():
    X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    # model prediction
    def forward(x):
        return w * x

    # loss (mean square error)
    def loss(y, y_predicted):
        return ((y_predicted - y) ** 2).mean()

    print(f'Prediction before training: f(5) = {forward(5):.3f}')

    # Training
    learning_rate = 0.01
    n_iterations = 100

    for epoch in range(n_iterations):
        # prediction = forward pass
        y_pred = forward(X)

        # loss
        l = loss(Y, y_pred)

        # gradients = backward pass
        l.backward()  # dl/dw

        # update weights
        with torch.no_grad():
            w -= learning_rate * w.grad

        # zero gradients
        w.grad.zero_()

        if epoch % 10 == 0:
            print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(5) = {forward(5):.3f}')


def torch_gradient_loss():
    X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    # model prediction
    def forward(x):
        return w * x

    print(f'Prediction before training: f(5) = {forward(5):.3f}')

    # Training
    learning_rate = 0.01
    n_iterations = 100

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([w], lr=learning_rate)

    for epoch in range(n_iterations):
        # prediction = forward pass
        y_pred = forward(X)

        # loss
        l = loss(Y, y_pred)

        # gradients = backward pass
        l.backward()  # dl/dw

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(5) = {forward(5):.3f}')


def torch_gradient_forward():
    X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

    X_test = torch.tensor([5], dtype=torch.float32)
    n_samples, n_features = X.shape
    print(n_samples, n_features)

    input_size = n_features
    output_size = n_features

    model = nn.Linear(input_size, output_size)

    print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

    # Training
    learning_rate = 0.02
    n_iterations = 1000

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_iterations):
        # prediction = forward pass
        y_pred = model(X)

        # loss
        l = loss(Y, y_pred)

        # gradients = backward pass
        l.backward()  # dl/dw

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, b] = model.parameters()
            print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


if __name__ == '__main__':
    # torch_gradient()
    # torch_gradient_loss()
    torch_gradient_forward()
