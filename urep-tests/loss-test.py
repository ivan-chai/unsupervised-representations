#!/usr/bin/env python3
import numpy as np

import torch

from urep import test
from urep.estimators.gcpc import get_maximal_mutual_information
from urep.loss import *
from urep.utils import to_tensor


class TestInfoNCELoss(test.TestCase):
    def test_gauss_comparator(self):
        batch_size = 4096
        dim = 4
        sigma2 = 0.7
        batch_x = np.zeros((batch_size, 2, dim), dtype=np.float32)
        batch_y = np.zeros((batch_size, 2, dim), dtype=np.float32)
        for i in range(batch_size):
            centroid = np.random.normal(scale=np.sqrt(sigma2), size=dim)
            batch_x[i, 1] = np.random.normal(loc=centroid, scale=np.sqrt(1 - sigma2), size=dim)
            batch_y[i, 0] = np.random.normal(loc=centroid, scale=np.sqrt(1 - sigma2), size=dim)
        mutual_information_gt = get_maximal_mutual_information(dim, sigma2)

        config = {
            "min_context_size": 1,
            "future_steps": 1,
            "comparator": "gauss",
            "comparator_params": {"init_centroid_sigma2": sigma2}
        }
        # Original InfoNCE.
        loss = InfoNCELoss(dim, dim, config)
        with torch.no_grad():
            loss_value = loss.forward(torch.from_numpy(batch_x), torch.from_numpy(batch_y)).numpy()
        self.assertEqualArrays(loss_value.shape, [batch_size])
        mutual_information = - np.mean(loss_value)
        self.assertAlmostEqual(mutual_information, mutual_information_gt, places=1)

        # Symmetric InfoNCE.
        config_sym = config.copy()
        config_sym["symmetric"] = True
        loss = InfoNCELoss(dim, dim, config_sym)
        with torch.no_grad():
            loss_value = loss.forward(torch.from_numpy(batch_x), torch.from_numpy(batch_y)).numpy()
        self.assertEqualArrays(loss_value.shape, [batch_size])
        mutual_information = - np.mean(loss_value)
        self.assertAlmostEqual(mutual_information, mutual_information_gt, places=1)

        # Transposed InfoNCE.
        config_trans = config.copy()
        config_trans["transpose"] = True
        loss = InfoNCELoss(dim, dim, config_trans)
        with torch.no_grad():
            loss_value = loss.forward(torch.from_numpy(batch_x), torch.from_numpy(batch_y)).numpy()
        self.assertEqualArrays(loss_value.shape, [batch_size])
        mutual_information = - np.mean(loss_value)
        self.assertAlmostEqual(mutual_information, mutual_information_gt, places=1)


class TestInfoNCEAggregatedLoss(test.TestCase):
    def test_compare_general_case(self):
        """Compare aggregated loss to general case InfoNCELoss."""
        self._cmp({"comparator": "gauss"})
        self._cmp({"comparator": "gauss", "symmetric": True})
        self._cmp({"comparator": "gauss", "transpose": True})

    def _cmp(self, config):
        batch_size = 8
        dim = 4
        batch1 = np.random.random((batch_size, dim))
        batch2 = np.random.random((batch_size, dim))
        agg_loss = InfoNCEArregatedLoss(dim, dim, config=config)
        agg_loss_value = agg_loss(to_tensor(batch1), to_tensor(batch2)).detach().numpy()

        # In timed loss first timestamp of the batch2 is compared with second timestamp of batch1.
        batch1_time = np.empty((batch_size, 2, dim))
        batch2_time = np.empty((batch_size, 2, dim))
        batch1_time[:, 1, :] = batch1
        batch2_time[:, 0, :] = batch2
        time_config = config.copy()
        time_config["min_context_size"] = 1
        time_config["future_steps"] = 1
        loss = InfoNCELoss(dim, dim, config=time_config)
        loss.comparator0 = agg_loss.comparator
        loss_value = loss(to_tensor(batch1_time), to_tensor(batch2_time)).detach().numpy()
        self.assertAlmostEqualArrays(agg_loss_value, loss_value)


if __name__ == "__main__":
    test.main()
