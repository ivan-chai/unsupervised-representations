"""Evaluation tools."""
import logging
from collections import OrderedDict

import torch
from torchnet.meter import AverageValueMeter

from . import logger
from .checkpoint import get_latest_checkpoint_step, read_checkpoint
from .config import prepare_config
from .data import make_dataloader
from .utils import to_tuple, try_cuda


class Evaluator(object):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("num_steps", None),
            ("batch_size", 32),
            ("num_workers", 8)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(config, self.get_default_config())

    @property
    def config(self):
        return self._config

    def evaluate(self, estimator, dataset):
        """Evaluate model on dataset."""
        logging.info("Start evaluation")
        data_loader = make_dataloader(dataset, self._config["batch_size"],
                                      num_steps=self._config["num_steps"],
                                      shuffle=True,
                                      num_workers=self._config["num_workers"])
        estimator = try_cuda(estimator)
        revert_training = estimator.training
        estimator.eval()
        loss_meter = AverageValueMeter()
        with torch.no_grad():
            for batch in logging.progress(data_loader):
                batch = to_tuple(batch)
                batch = [try_cuda(tensor) for tensor in batch]
                loss_value = estimator(*batch, compute_loss=True)["loss"]
                loss_meter.add(loss_value.item())
        output_string = "Evaluation finished"
        output_string += ", loss {:.5f}".format(loss_meter.value()[0])
        logging.info(output_string)
        if revert_training:
            estimator.train()

    def evaluation_hook(self, estimator, model_dir, dataset):
        """Load and evaluate model."""
        step = get_latest_checkpoint_step(model_dir)
        if step is None:
            raise RuntimeError("No checkpoints were detected in {}".format(model_dir))
        logging.info("Load state from step {}".format(step))
        estimator.load_state_dict(read_checkpoint(model_dir, step))
        self.evaluate(estimator, dataset)
