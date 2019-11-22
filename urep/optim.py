"""Optimizer wrappers."""
from collections import OrderedDict

import torch

from .config import prepare_config


class AdamOptimizer(torch.optim.Adam):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("learning_rate", 1e-4),
            ("weight_decay", 1e-4),
            ("gradients_clip", 5.0)
        ])

    def __init__(self, model, config=None):
        self._config = prepare_config(config, self.get_default_config())
        super().__init__(model.parameters(),
                         lr=self._config["learning_rate"],
                         weight_decay=self._config["weight_decay"])
        self._model = model

    def step(self):
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config["gradients_clip"])
        return super().step()


OPTIMIZERS = {
    "adam": AdamOptimizer
}
