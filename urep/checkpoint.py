"""Tools for working with checkpoints."""
import os

import torch
from .io import ensure_directory


CHECKPOINT_FORMAT = "{step:010d}.pth.tar"


def get_latest_checkpoint_step(model_dir):
    """Returns step id of the latest checkpoint."""
    if not os.path.exists(model_dir):
        return None
    max_step = -1
    for filename in os.listdir(model_dir):
        ext = filename[filename.find("."):]
        if ext != ".pth.tar":
            continue
        noext = filename[:filename.find(".")]
        step = int(noext)
        max_step = max(max_step, step)
    return max_step if max_step >= 0 else None


def read_checkpoint(model_dir, step=None):
    """Load checkpoint from model dir. If step is None, load latest checkpoint."""
    step = get_latest_checkpoint_step(model_dir) if step is None else step
    if step is None:
        raise FileNotFoundError("No checkpoints in {}".format(model_dir))
    filename = CHECKPOINT_FORMAT.format(step=step)
    path = os.path.join(model_dir, filename)
    state_dict = torch.load(path)
    return state_dict


def write_checkpoint(state_dict, model_dir, step):
    """Dump torch state dict to the model dir."""
    ensure_directory(model_dir)
    filename = CHECKPOINT_FORMAT.format(step=step)
    path = os.path.join(model_dir, filename)
    torch.save(state_dict, path)
