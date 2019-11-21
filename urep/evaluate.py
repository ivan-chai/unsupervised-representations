"""Evaluation tools."""
import logging
import math
import random
from collections import OrderedDict

import torch
from torchnet.meter import AverageValueMeter

from . import logger
from .checkpoint import get_latest_checkpoint_step, read_checkpoint
from .config import prepare_config
from .data import make_dataloader
from .loss import LOSSES
from .utils import to_tuple, try_cuda


class Evaluator(object):
    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("batch_size", 32),
            ("num_steps", None),
            ("num_workers", 8),
            ("loss", "info_nce"),
            ("loss_params", None)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(config, self.get_default_config())

    @property
    def config(self):
        return self._config

    def evaluate(self, model, dataset):
        """Evaluate model on dataset."""
        data_loader = make_dataloader(dataset, self._config["batch_size"],
                                      num_steps=self._config["num_steps"],
                                      shuffle=True,
                                      num_workers=self._config["num_workers"])
        loss = LOSSES[self._config["loss"]](self._config["loss_params"])
        loss_meter = AverageValueMeter()
        data_names = to_tuple(dataset[0].data_names)
        model_input_names = to_tuple(model.input_names)
        model_output_names = to_tuple(model.output_names)
        loss_input_names = to_tuple(loss.input_names)
        with torch.no_grad():
            for batch in logging.progress(data_loader):
                batch = to_tuple(batch)
                if len(batch) != len(data_names):
                    raise ValueError("Length mismatch between batch and dataset names: {} != {}".format(
                        len(batch), len(data_names)))
                if torch.cuda.is_available():
                    batch = [try_cuda(tensor) for tensor in batch]
                named_batch = OrderedDict([(name, tensor) for name, tensor in zip(data_names, batch)])
                input_batch = OrderedDict([(name, named_batch[name]) for name in model.input_names])
                result = model(*list(input_batch.values()))
                if len(result) != len(model_output_names):
                    raise ValueError("Length mismatch between model result tensors and model output names: {} != {}".format(
                        len(result), len(model_output_names)))
                named_result = OrderedDict([(name, tensor) for name, tensor in zip(model.output_names, result)])
                named_io = named_batch.copy()
                named_io.update(named_result)
                loss_input = OrderedDict([(name, named_io[name]) for name in loss.input_names])
                loss_value = self._loss(*list(loss_input.values()))
                loss_meter.add(loss_value.item())
        output_string = "Evaluation finished"
        output_string += ", loss {:.5f}".format(loss_meter.value()[0])
        logging.info(output_string)

    def evaluation_hook(self, model, model_dir, dataset):
        """Load and evaluate model."""
        step = get_latest_checkpoint_step(model_dir)
        if step is None:
            raise RuntimeError("No checkpoints were detected in {}".format(model_dir))
        logging.info("Load state from step {}".format(step))
        model.load_state_dict(read_checkpoint(model_dir, step))
        model.eval()
        model = try_cuda(model)
        self.evaluate(model, dataset)
