"""Convolutional network for audio processing."""
from collections import OrderedDict
import torch

from ..config import prepare_config


class FullyConnected(torch.nn.Module):
    """Simple network of Fully Connected layers."""
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("num_channels", [512] * 5)
        ])

    def __init__(self, in_channels, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        num_layers = len(self._config["num_channels"])
        layers = []
        for i, out_channels in enumerate(self._config["num_channels"]):
            layers.append(torch.nn.Linear(in_channels, out_channels))
            if i < num_layers - 1:
                layers.append(torch.nn.ReLU())
            in_channels = out_channels
        self.body = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Apply model to input tensor x with shape (batch, *, dim)."""
        if len(x.shape) < 2:
            raise ValueError("Expected tensor of shape (batch, *, dim), got: {}".format(
                x.shape))
        out = self.body(x)
        return out

    @property
    def out_channels(self):
        return self._config["num_channels"][-1]
