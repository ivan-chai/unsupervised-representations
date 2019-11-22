"""Training tools."""
import itertools
import logging
from collections import OrderedDict

import torch

from .checkpoint import get_latest_checkpoint_step, read_checkpoint, write_checkpoint
from .config import prepare_config
from .data import make_dataloader
from .io import ensure_directory
from .loss import LOSSES
from .optim import OPTIMIZERS
from .utils import to_tuple, try_cuda


class Trainer(object):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("num_steps", 5000),
            ("batch_size", 128),
            ("num_workers", 8),
            ("logging_steps", 100),
            ("checkpoint_steps", 500),
            ("optimizer", "adam"),
            ("optimizer_params", None)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(config, self.get_default_config())

    @property
    def config(self):
        return self._config

    def train(self, estimator, dataset, model_dir, eval_hook=None, collate_fn=None):
        """Train model."""
        eval_hook = eval_hook if eval_hook is not None else lambda: None
        data_loader = make_dataloader(dataset, self._config["batch_size"],
                                      shuffle=True,
                                      num_workers=self._config["num_workers"])
        estimator = try_cuda(estimator)
        estimator.train()
        initial_step = get_latest_checkpoint_step(model_dir)
        if initial_step is None:
            initial_step = 0
        else:
            logging.info("Load state from step {}".format(initial_step))
            estimator.load_state_dict(read_checkpoint(model_dir, initial_step))
        optimizer = OPTIMIZERS[self._config["optimizer"]](estimator, self._config["optimizer_params"])
        steps = range(initial_step, self._config["num_steps"])
        for step, batch in zip(steps, itertools.cycle(data_loader)):
            batch = to_tuple(batch)
            if torch.cuda.is_available():
                batch = [try_cuda(tensor) for tensor in batch]
            loss_value = estimator(*batch, compute_loss=True)["loss"]
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if step % self._config["logging_steps"] == 0:
                logging.info("Step {}, train loss {}".format(step, loss_value.item()))

            if (step % self._config["checkpoint_steps"] == 0) or (step == steps[-1]):
                logging.info("Dump checkpoint for step {}".format(step))
                write_checkpoint(estimator.state_dict(), model_dir, step)
                eval_hook()
