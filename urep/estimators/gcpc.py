"""Model and tools for Gaussian Contrastive Predictive Coding."""
from collections import OrderedDict

import numpy as np
import torch

from ..config import prepare_config
from ..loss import LOSSES
from ..models import AGGREGATORS, MODELS


class GCPCModel(torch.nn.Module):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("model", "audio_cnn"),
            ("model_params", None)
        ])

    def __init__(self, in_channels, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self.encoder = MODELS[self._config["model"]](in_channels, self._config["model_params"])

    @property
    def embedding_size(self):
        return self.encoder.out_channels

    def forward(self, batch):
        embeddings = self.encoder(batch)
        return embeddings


class GCPCEstimator(torch.nn.Module):
    """Class encapsulates model, loss and metrics."""
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("model_params", None),
            ("loss", "info_nce"),
            ("loss_params", None)
        ])

    def __init__(self, in_channels, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self.model = GCPCModel(in_channels, self._config["model_params"])
        loss_class = LOSSES[self._config["loss"]]
        self.loss = loss_class(self.model.embedding_size, self.model.embedding_size,
                               config=self._config["loss_params"])

    def forward(self, waveforms, labels=None, compute_loss=False):
        embeddings = self.model(waveforms)
        result = {"embeddings": embeddings}
        if compute_loss:
            loss_value = self.loss(embeddings, embeddings)
            result["loss"] = loss_value
        return result

    def state_dict(self):
        return {"state_dict": self.model.state_dict(),
                "loss_state": self.loss.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["state_dict"])
        self.loss.load_state_dict(state_dict["loss_state"])


def get_maximal_mutual_information(dim, centroid_sigma2):
    return - (dim / 2) * np.log(1 - centroid_sigma2 ** 2)
