"""Layers for embedding comparison."""
import numpy as np
import torch


class BilinearComparator(torch.nn.Module):
    def __init__(self, x_dim, y_dim, config=None):
        super().__init__()
        self.transform = torch.nn.Linear(x_dim, y_dim, bias=False)

    def forward(self, x, y):
        """Embeddings should be (batch, time, dim) or (batch, dim)."""
        if len(x.shape) != len(y.shape):
            raise ValueError("Input rank mismatch: {} != {}".format(
                len(x.shape), len(y.shape)))
        has_time = (len(x.shape) == 3)
        if has_time:
            if x.shape[1] != y.shape[1]:
                raise ValueError("Time dimension mismatch: {} != {}".format(
                    x.shape[1], y.shape[1]))
        input_x_shape = x.shape
        input_y_shape = y.shape
        x = self.transform(x).reshape(-1, input_y_shape[-1])
        y = y.reshape(-1, input_y_shape[-1])
        out = torch.matmul(x, y.t())  # (batch, batch) or (batch x time, batch x time).
        if has_time:
            batch_size, duration = input_x_shape[:2]
            out = out.view(batch_size, duration, batch_size, duration)
        return out


class Gaussian(torch.nn.Module):
    def __init__(self, x_dim, y_dim, config=None):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.from_numpy(np.zeros([])))

    def forward(self, x, y):
        """Embeddings should be (batch, time, dim) or (batch, dim)."""
        if x.shape != y.shape:
            raise ValueError("Input shape mismatch: {} != {}".format(
                x.shape, y.shape))
        input_shape = x.shape
        has_time = (len(input_shape) == 3)
        if has_time:
            dim = input_shape[-1]
            x = x.reshape(-1, dim)
            y = y.reshape(-1, dim)
        sigma2 = torch.sigmoid(self.alpha)
        sigma4 = sigma2 * sigma2
        x2 = (x * x).sum(-1)  # (batch).
        y2 = (y * y).sum(-1)  # (batch).
        c = -0.5 * dim * torch.log(1 - sigma4)
        c_sum = -0.5 * sigma4 / (1 - sigma4)
        c_prod = sigma2 / (1 - sigma4)
        out = c + c_prod * torch.matmul(x, y.t()) + c_sum * (x2.unsqueeze(1) + y2.unsqueeze(0)) # (batch, batch) or (batch x time, batch x time).
        if has_time:
            batch_size, duration = input_shape[:2]
            out = out.view(batch_size, duration, batch_size, duration)
        return out


COMPARATORS = {
    "bilinear": BilinearComparator,
    "gauss": Gaussian
}
