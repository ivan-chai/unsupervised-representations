"""Optimizer wrappers."""
from collections import OrderedDict

import torch

from .config import prepare_config


class AdamOptimizer(object):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("learning_rate", 3e-4),
            ("weight_decay", 1e-4)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(config, self.get_default_config())

    def __call__(self, model):
        return torch.optim.Adam(model.parameters(),
                                lr=self._config["learning_rate"],
                                weight_decay=self._config["learning_rate"])


OPTIMIZERS = {
    "adam": AdamOptimizer
}
