"""Losses for MMI training."""
from collections import OrderedDict

import numpy as np
import torch

from .config import prepare_config
from .comparator import BilinearComparator


class NamedLoss(torch.nn.Module):
    input_names = None


class InfoNCELoss(NamedLoss):
    input_names = ("embeddings", "contexts")

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("min_context_size", 64),
            ("future_steps", 12),
            ("comparator_params", None)
        ])

    def __init__(self, embeddings_size, contexts_size, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        self._comparator = BilinearComparator(embeddings_size, contexts_size,
                                              config=self._config["comparator_params"])

    def forward(self, embeddings, contexts):
        if len(embeddings.shape) != 3:
            raise ValueError("Expected embeddings with shape (batch, time, dim")
        if len(contexts.shape) != 3:
            raise ValueError("Expected contexts with shape (batch, time, dim")
        if embeddings.shape[:2] != contexts.shape[:2]:
            raise ValueError("Embeddings and contexts shape mismatch: {} != {}".format(
                embeddings.shape[:2], contexts.shape[:2]))
        batch_size, duration = embeddings.shape[:2]
        min_context_size = self._config["min_context_size"]
        future_steps = self._config["future_steps"]
        if embeddings.shape[1] != contexts.shape[1]:
            raise ValueError("Features and contexts duration mismatch")
        if embeddings.shape[1] < min_context_size + future_steps:
            raise ValueError("Duration is not enough for InfoNCE loss")
        flat_log_probabilities_array = []
        for step in range(future_steps):
            embeddings_subset = embeddings[:, min_context_size + step:duration]
            contexts_subset = contexts[:, min_context_size - 1:duration - step - 1]
            subset_duration = duration - step - min_context_size
            log_density_ratios_matrix = self._comparator(embeddings_subset, contexts_subset)
            log_density_ratios_positive = log_density_ratios_matrix.reshape(batch_size * subset_duration, batch_size * subset_duration).diag().view(batch_size, subset_duration)
            log_density_ratios_matrix_half_flat = log_density_ratios_matrix.view(batch_size, subset_duration, batch_size * subset_duration)  # (batch, time, batch x time).
            log_density_ratio_sums = torch.logsumexp(log_density_ratios_matrix_half_flat, dim=-1)  # (batch, time).
            # We implement mean instead of sum for better loss values. It is not part of original approach.
            log_density_ratio_sums = log_density_ratio_sums - np.log(batch_size * subset_duration)
            log_probabilities = log_density_ratios_positive - log_density_ratio_sums  # (batch, time).
            flat_log_probabilities_array.append(log_probabilities.flatten())  # (batch x time).
        flat_log_probabilities = torch.cat(flat_log_probabilities_array, dim=0)
        total_loss = -flat_log_probabilities.mean()
        return total_loss


class InfoNCELossArregated(NamedLoss):
    @staticmethod
    def get_default_config():
        return OrderedDict([
        ])

    def forward(self, log_density_ratios):
        """Simple version without time."""
        input_shape = log_density_ratios.shape
        if (len(input_shape) != 2) or (input_shape[0] != input_shape[1]):
            raise ValueError("Expected square matrix with shape (batch, batch), got {}".format(list(input_shape)))
        log_density_ratios_positive = torch.diag(log_density_ratios)
        log_density_ratio_sums = torch.logsumexp(log_density_ratios_matrix, axis=-1)  # (batch).
        # We implement mean instead of sum for better loss values. It is not part of original approach.
        batch_size = input_shape[0]
        log_density_ratio_sums -= np.log(batch_size)
        log_probabilities = log_density_ratios_positive - log_density_ratio_sums  # (batch).
        total_loss = -log_probabilities.mean()
        return total_loss

    @property
    def input_names(self):
        return "log_density_ratios"


LOSSES = {
    "info_nce": InfoNCELoss,
    "info_nce_aggregated": InfoNCELossArregated
}
