#!/usr/bin/env python3
import numpy as np

import torch

from urep import test
from urep.estimators.gcpc import get_maximal_mutual_information
from urep.loss import *


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


if __name__ == "__main__":
    test.main()
