"""High level interface to the model."""
from collections import OrderedDict

import torch

from .comparator import BilinearComparator
from .config import prepare_config
from .models import AGGREGATORS, MODELS


class NamedModule(torch.nn.Module):
    @property
    def input_names(self):
        raise NotImplementedError("Base class method is not implemented")

    @property
    def output_names(self):
        raise NotImplementedError("Base class method is not implemented")

    @property
    def output_loss_name(self):
        raise NotImplementedError("Base class method is not implemented")


class CPCModel(NamedModule):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("embedding_size", 512),
            ("context_size", 512),
            ("model", "audio_cnn"),
            ("model_params", None),
            ("aggregator", "gru"),
            ("aggregator_params", None)
        ])

    def __init__(self, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self._model = MODELS[self._config["model"]](self._config["model_params"])
        self._aggregator = AGGREGATORS[self._config["aggregator"]](self._config["embedding_size"], self._config["aggregator_params"])

    def forward(self, waveform):
        if len(waveform.shape) != 2:
            raise ValueError("Expected tensor of shape (batch, time)")
        embeddings = self._model(waveform)
        contexts = self._aggregator(embeddings)
        return embeddings, contexts

    @property
    def input_names(self):
        return "waveform"

    @property
    def output_names(self):
        return "embeddings", "contexts"

    @property
    def output_dims(self):
        return self._model.output_dims, self._aggregator.output_dims
