"""RNN for context aggregation."""
from collections import OrderedDict
import torch

from ..config import prepare_config
from .base import ModelBase


class GRUAggregator(ModelBase):
    """Simple LSTM aggregator similar to that from Contrastive Predictive
    Coding (CPC).
    """
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("num_channels", 256),
        ])

    def __init__(self, in_channels, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self.rnn = torch.nn.GRU(in_channels, self._config["num_channels"])

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError("Expected tensor of shape (batch, time, dim)")
        x = x.permute(1, 0, 2)
        out, _ = self.rnn(x)
        out = out.permute(1, 0, 2)
        return out

    @property
    def output_dims(self):
        return self._config["num_channels"]
