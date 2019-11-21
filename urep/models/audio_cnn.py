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
        
    def __init__(self, config=None):
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
            in_channels = out_channels
        self.body = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)
