"""Layers for embedding comparison."""
from collections import OrderedDict

import numpy as np
import torch

from .config import prepare_config


class BilinearComparator(torch.nn.Module):
    def __init__(self, x_dim, y_dim, config=None):
        super().__init__()
        if config:
            raise ValueError("Config should be empty")
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


class GaussianComparator(torch.nn.Module):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("init_centroid_sigma2", 0.5),
            ("trainable", True)
        ])

    def __init__(self, x_dim, y_dim, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        init_sigma2 = self._config["init_centroid_sigma2"]
        init_alpha = np.log(init_sigma2 / (1 - init_sigma2))
        self.alpha = torch.nn.Parameter(torch.from_numpy(np.array(init_alpha, dtype=np.float32)),
                                        requires_grad=self._config["trainable"])

    def forward(self, x, y):
        """Embeddings should be (batch, time, dim) or (batch, dim)."""
        if x.shape != y.shape:
            raise ValueError("Input shape mismatch: {} != {}".format(
                x.shape, y.shape))
        input_shape = x.shape
        has_time = (len(input_shape) == 3)
        dim = input_shape[-1]
        if has_time:
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
    "gauss": GaussianComparator
}
