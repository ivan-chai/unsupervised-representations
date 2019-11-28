"""Convolutional network for audio processing."""
from collections import OrderedDict
import torch

from ..config import prepare_config


class AudioCNN(torch.nn.Module):
    """Simple network with 1-D convolutions, similar to that from
    Contrastive Predictive Coding (CPC)."""
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("num_channels", [512] * 5),
            ("kernel_sizes", [10, 8, 4, 4, 4]),
            ("strides", [5, 4, 2, 2, 2])
        ])

    def __init__(self, in_channels=1, config=None):
        if in_channels != 1:
            raise ValueError("AuidoCNN should be applied to 1-D input")
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        num_layers = len(self._config["num_channels"])
        if len(self._config["kernel_sizes"]) != num_layers:
            raise ValueError("Wrong number of kernel sizes: {} != {}".format(
                len(self._config["kernel_sizes"]), num_layers))
        if len(self._config["strides"]) != num_layers:
            raise ValueError("Wrong number of strides: {} != {}".format(
                len(self._config["strides"]), num_layers))
        layers = []
        in_channels = 1
        for i, out_channels in enumerate(self._config["num_channels"]):
            stride = self._config["strides"][i]
            kernel_size = self._config["kernel_sizes"][i]
            layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride))
            if i < num_layers - 1:
                layers.append(torch.nn.ReLU())
            in_channels = out_channels
        self.body = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Apply model to input tensor x with shape (batch, time, 1)."""
        if (len(x.shape) != 3) or (x.shape[-1] != 1):
            raise ValueError("Expected tensor of shape (batch, time, 1)")
        x = x.squeeze(-1)  # (batch, time).
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = x.unsqueeze(1)  # (batch, 1, time).
        out = self.body(x)
        out = out.permute(0, 2, 1)
        return out

    @property
    def out_channels(self):
        return self._config["num_channels"][-1]
