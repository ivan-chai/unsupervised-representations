#!/usr/bin/env python3
import numpy as np

import torch

from urep import test
from urep.comparator import *


class TestGaussianComparator(test.TestCase):
    def test_simple(self):
        batch_size = 4096
        dim = 4
        sigma2 = 0.7
        batch_x = np.zeros((batch_size, dim), dtype=np.float32)
        batch_y = np.zeros((batch_size, dim), dtype=np.float32)
        for i in range(batch_size):
            centroid = np.random.normal(scale=np.sqrt(sigma2), size=dim)
            batch_x[i] = np.random.normal(loc=centroid, scale=np.sqrt(1 - sigma2), size=dim)
            batch_y[i] = np.random.normal(loc=centroid, scale=np.sqrt(1 - sigma2), size=dim)
            
        config = {"init_centroid_sigma2": sigma2}
        comparator = GaussianComparator(dim, dim, config)
        with torch.no_grad():
            cmp_mat = comparator.forward(torch.from_numpy(batch_x), torch.from_numpy(batch_y)).numpy()
        self.assertEqualArrays(cmp_mat.shape, [batch_size, batch_size])
        mutual_information = np.mean(np.diag(cmp_mat))
        mutual_information_gt = - (dim / 2) * np.log(1 - sigma2 ** 2)
        self.assertAlmostEqual(mutual_information, mutual_information_gt, places=1)


if __name__ == "__main__":
    test.main()
