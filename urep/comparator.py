"""Layers for embedding comparison."""
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
        x = self.transform(x).view(-1, input_y_shape[-1])
        y = y.reshape(-1, input_y_shape[-1])
        out = torch.matmul(x, y.t())  # (batch, batch) or (batch x time, batch x time).
        if has_time:
            batch_size, duration = input_x_shape[:2]
            out = out.view(batch_size, duration, batch_size, duration)
        return out
