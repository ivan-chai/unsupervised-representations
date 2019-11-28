"""Losses for MMI training."""
from collections import OrderedDict

import numpy as np
import torch

from .config import prepare_config
from .comparator import COMPARATORS


class NamedLoss(torch.nn.Module):
    input_names = None


class InfoNCELoss(NamedLoss):
    input_names = ("embeddings", "contexts")

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("min_context_size", 64),
            ("future_steps", 12),
            ("single_comparator", False),
            ("transpose", False),
            ("symmetric", False),
            ("comparator", "bilinear"),
            ("comparator_params", None)
        ])

    def __init__(self, embeddings_size, contexts_size, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        for step in range(self._config["future_steps"] if not self._config["single_comparator"] else 1):
            comparator_class = COMPARATORS[self._config["comparator"]]
            comparator = comparator_class(embeddings_size, contexts_size,
                                          config=self._config["comparator_params"])
            self.add_module("comparator{}".format(step), comparator)

    def forward(self, embeddings, contexts):
        transpose = self._config["transpose"]
        loss = self._info_nce(embeddings, contexts, transpose=transpose)
        if self._config["symmetric"]:
            loss = 0.5 * (loss + self._info_nce(embeddings, contexts, transpose=not transpose))
        return loss

    def _info_nce(self, embeddings, contexts, transpose=False):
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
            comparator = self._modules["comparator{}".format(step if not self._config["single_comparator"] else 0)]
            log_density_ratios_matrix = comparator(embeddings_subset, contexts_subset)  # (batch, time, batch, time).
            if transpose:
                log_density_ratios_matrix = log_density_ratios_matrix.permute(2, 3, 0, 1)
            # Numerator consists from ratios for matching (batch, time) pairs.
            log_density_ratios_positive = log_density_ratios_matrix.view(batch_size * subset_duration, batch_size * subset_duration).diag().view(batch_size, subset_duration)
            # Negatives are obtained from different batch elements for the same time step.
            # Denumerator is just a sum of ratios for diferent samples from the batch.
            log_density_ratios_alt = torch.diagonal(log_density_ratios_matrix, dim1=1, dim2=3)  # (batch, batch, time).
            log_density_ratio_sums = torch.logsumexp(log_density_ratios_alt, dim=1)  # (batch, time).
            # We implement mean instead of sum for better loss values. It is not part of original approach.
            log_density_ratio_sums = log_density_ratio_sums - np.log(batch_size)
            log_probabilities = log_density_ratios_positive - log_density_ratio_sums  # (batch, time).
            flat_log_probabilities_array.append(log_probabilities.flatten())  # (batch x time).
        flat_log_probabilities = torch.cat(flat_log_probabilities_array, dim=0)
        losses = -flat_log_probabilities
        return losses


class InfoNCEArregatedLoss(NamedLoss):
    """Simple version of InfoNCE without time dimension."""
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("transpose", False),
            ("symmetric", False),
            ("comparator", "bilinear"),
            ("comparator_params", None)
        ])

    def __init__(self, embeddings_size, contexts_size, config=None):
        super().__init__()
        self._config = prepare_config(config, self.get_default_config())
        comparator_class = COMPARATORS[self._config["comparator"]]
        self.comparator = comparator_class(embeddings_size, contexts_size,
                                           config=self._config["comparator_params"])

    def forward(self, embeddings, contexts):
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings shape should be (batch, dim), got: {}".format(
                embeddings.shape))
        if len(contexts.shape) != 2:
            raise ValueError("Contexts shape should be (batch, dim), got: {}".format(
                contexts.shape))
        transpose = self._config["transpose"]
        loss = self._info_nce(embeddings, contexts, transpose=transpose)
        if self._config["symmetric"]:
            loss = 0.5 * (loss + self._info_nce(embeddings, contexts, transpose=not transpose))
        return loss

    def _info_nce(self, embeddings, contexts, transpose=False):
        log_density_ratios_matrix = self.comparator(embeddings, contexts)  # (batch, batch).
        if transpose:
            log_density_ratios_matrix = log_density_ratios_matrix.permute(1, 0)
        log_density_ratios_positive = torch.diag(log_density_ratios_matrix)
        log_density_ratio_sums = torch.logsumexp(log_density_ratios_matrix, axis=-1)  # (batch).
        # We implement mean instead of sum for better loss values. It is not part of original approach.
        batch_size = embeddings.shape[0]
        log_density_ratio_sums = log_density_ratio_sums - np.log(batch_size)
        log_probabilities = log_density_ratios_positive - log_density_ratio_sums  # (batch).
        losses = -log_probabilities
        return losses


LOSSES = {
    "info_nce": InfoNCELoss,
    "info_nce_aggregated": InfoNCEArregatedLoss
}
