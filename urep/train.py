"""Training tools."""
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
            ("batch_size", 32),
            ("num_workers", 8),
            ("logging_steps", 100),
            ("checkpoint_steps", 500),
            ("optimizer", "adam"),
            ("optimizer_params", None),
            ("loss", "info_nce"),
            ("loss_params", None)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(config, self.get_default_config())
        self._optimizer = OPTIMIZERS[self._config["optimizer"]](self._config["optimizer_params"])

    @property
    def config(self):
        return self._config

    def train(self, model, dataset, model_dir, eval_hook=None, collate_fn=None):
        """Train model."""
        eval_hook = eval_hook if eval_hook is not None else lambda: None
        data_loader = make_dataloader(dataset, self._config["batch_size"],
                                      shuffle=True,
                                      num_workers=self._config["num_workers"])
        model = try_cuda(model)
        model_input_names = to_tuple(model.input_names)
        model_output_names = to_tuple(model.output_names)
        output_dims_dict = dict(zip(model_output_names, model.output_dims))
        loss_class = LOSSES[self._config["loss"]]
        loss_input_names = to_tuple(loss_class.input_names)
        loss_input_dims = [output_dims_dict[name] for name in loss_input_names]
        loss = loss_class(*loss_input_dims, self._config["loss_params"])
        model.train()
        loss.train()
        initial_step = get_latest_checkpoint_step(model_dir)
        if initial_step is None:
            initial_step = 0
        else:
            logging.info("Load state from step {}".format(initial_step))
            model.load_state_dict(read_checkpoint(model_dir, initial_step)["state_dict"])
            loss.load_state_dict(read_checkpoint(model_dir, initial_step)["loss"])
        optimizer = self._optimizer(model)
        data_names = to_tuple(dataset[0].data_names)
        steps = range(initial_step, self._config["num_steps"])
        for step, batch in zip(steps, data_loader):
            batch = to_tuple(batch)
            if len(batch) != len(data_names):
                raise ValueError("Length mismatch between batch and dataset names: {} != {}".format(
                    len(batch), len(data_names)))
            if torch.cuda.is_available():
                batch = [try_cuda(tensor) for tensor in batch]
            named_batch = OrderedDict([(name, tensor) for name, tensor in zip(data_names, batch)])
            input_batch = OrderedDict([(name, named_batch[name]) for name in model_input_names])
            for tensor in input_batch.values():
                tensor.requires_grad = True
            optimizer.zero_grad()
            result = model(*list(input_batch.values()))
            if len(result) != len(model_output_names):
                raise ValueError("Length mismatch between model result tensors and model output names: {} != {}".format(
                    len(result), len(model_output_names)))
            named_result = OrderedDict([(name, tensor) for name, tensor in zip(model_output_names, result)])
            named_io = named_batch.copy()
            named_io.update(named_result)
            loss_input = OrderedDict([(name, named_io[name]) for name in loss_input_names])
            loss_value = loss(*list(loss_input.values()))
            loss_value.backward()
            optimizer.step()

            if step % self._config["logging_steps"] == 0:
                logging.info("Step {}, train loss {}".format(step, loss_value.item()))

            if (step % self._config["checkpoint_steps"] == 0) or (step == steps[-1]):
                logging.info("Dump checkpoint for step {}".format(step))
                state_dict = OrderedDict([
                    ("state_dict", model.state_dict()),
                    ("loss", loss.state_dict())
                ])
                write_checkpoint(state_dict, model_dir, step)
                eval_hook()
