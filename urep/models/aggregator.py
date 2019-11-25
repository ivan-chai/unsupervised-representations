"""RNN for context aggregation."""
from collections import OrderedDict
import logging
import torch

from ..config import prepare_config
from ..parallel import SuppressOutput
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
        self.rnn = torch.nn.GRU(in_channels, self._config["num_channels"], batch_first=True)
        logging.warning("Supress RNN Dataparallel warnings in forward")

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError("Expected tensor of shape (batch, time, dim)")
        with SuppressOutput(suppress_stderr=True):
            out, _ = self.rnn(x)
        return out

    @property
    def output_dims(self):
        return self._config["num_channels"]
