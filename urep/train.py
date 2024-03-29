"""Training tools."""
import itertools
import logging
import os
from collections import OrderedDict

import torch
from tensorboardX import SummaryWriter

from .checkpoint import get_latest_checkpoint_step, read_checkpoint, write_checkpoint
from .config import prepare_config
from .data import make_dataloader
from .io import ensure_directory
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
        logging.info("Start training")
        logdir_train = os.path.join(model_dir, "logdir", "train")
        logdir_eval = os.path.join(model_dir, "logdir", "eval")
        ensure_directory(logdir_train)
        ensure_directory(logdir_eval)
        summary_train = SummaryWriter(logdir_train)
        summary_eval = SummaryWriter(logdir_eval)
        eval_hook = eval_hook if eval_hook is not None else lambda: None
        data_loader = make_dataloader(dataset, self._config["batch_size"],
                                      shuffle=True,
                                      num_workers=self._config["num_workers"])
        estimator = try_cuda(estimator)
        revert_eval = not estimator.training
        estimator.train()
        optimizer = OPTIMIZERS[self._config["optimizer"]](estimator, self._config["optimizer_params"])
        initial_step = get_latest_checkpoint_step(model_dir)
        if initial_step is None:
            initial_step = 0
        else:
            logging.info("Load state from step {}".format(initial_step))
            state_dict = read_checkpoint(model_dir, initial_step)
            if "optimizer" in state_dict:
                logging.info("Load optimizer state")
                optimizer.load_state_dict(state_dict.pop("optimizer"))
            else:
                logging.info("Optimizer state was not found in checkpoint")
            logging.info("Load model state")
            estimator.load_state_dict(state_dict)
        steps = range(initial_step + 1, self._config["num_steps"])
        for step, batch in zip(steps, itertools.cycle(data_loader)):
            batch = to_tuple(batch)
            batch = [try_cuda(tensor) for tensor in batch]
            loss_value = estimator(*batch, compute_loss=True)["loss"].mean()
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            summary_train.add_scalar("Loss", loss_value.item(), step)

            relative_step = step - initial_step - 1
            if relative_step % self._config["logging_steps"] == 0:
                logging.info("Step {}, train loss {}".format(step, loss_value.item()))

            if (relative_step % self._config["checkpoint_steps"] == 0) or (step == steps[-1]):
                logging.info("Dump checkpoint for step {}".format(step))
                state_dict = estimator.state_dict()
                state_dict["optimizer"] = optimizer.state_dict()
                write_checkpoint(state_dict, model_dir, step)
                eval_loss_value = eval_hook()
                if eval_loss_value is not None:
                    summary_eval.add_scalar("Loss", eval_loss_value, step)
        if revert_eval:
            estimator.eval()
