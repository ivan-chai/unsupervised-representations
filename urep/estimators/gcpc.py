"""Model and tools for Gaussian Contrastive Predictive Coding."""
from collections import OrderedDict

import torch

from ..config import prepare_config
from ..loss import LOSSES
from ..models import AGGREGATORS, MODELS


class GCPCModel(torch.nn.Module):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("model", "audio_cnn"),
            ("model_params", None),
            ("aggregator", "gru"),
            ("aggregator_params", None)
        ])

    def __init__(self, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self.encoder = MODELS[self._config["model"]](self._config["model_params"])
        aggregator_config = prepare_config(self._config["aggregator_params"], {"num_channels": self.encoder.output_dims})
        self.aggregator = AGGREGATORS[self._config["aggregator"]](self.encoder.output_dims, aggregator_config)
        if self.encoder.output_dims != self.aggregator.output_dims:
            raise ValueError("In G-CPC model aggregator and encoder should produce vectors of equal size")

    @property
    def embedding_size(self):
        return self.encoder.output_dims

    def forward(self, waveform):
        if len(waveform.shape) != 2:
            raise ValueError("Expected tensor of shape (batch, time)")
        embeddings = self.encoder(waveform)
        contexts = self.aggregator(embeddings)
        return embeddings, contexts


class GCPCEstimator(torch.nn.Module):
    """Class encapsulates model, loss and metrics."""
    @staticmethod
    def get_default_loss_config():
        return OrderedDict([
            ("comparator", "gauss"),
            ("single_comparator", True),
            ("symmetric", True)
        ])

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("model_params", None),
            ("loss_params", None)
        ])

    def __init__(self, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self.model = GCPCModel(self._config["model_params"])
        loss_config = prepare_config(self._config["loss_params"], self.get_default_loss_config())
        self.loss = LOSSES["info_nce"](self.model.embedding_size, self.model.embedding_size,
                                       config=loss_config)

    def forward(self, waveforms, labels=None, compute_loss=False):
        embeddings, contexts = self.model(waveforms)
        result = {"embeddings": embeddings,
                  "contexts": contexts}
        if compute_loss:
            loss_value = self.loss(embeddings, contexts)
            result["loss"] = loss_value
        return result

    def state_dict(self):
        return {"state_dict": self.model.state_dict(),
                "loss_state": self.loss.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["state_dict"])
        self.loss.load_state_dict(state_dict["loss_state"])
