"""Helper tools."""
import os
import sys

import torch


def to_tuple(obj):
    """Convert sequence to tuple or single element to tuple of length 1."""
    if isinstance(obj, tuple):
        return obj
    if isinstance(obj, list):
        return tuple(obj)
    return (obj, )


def try_cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()
    return obj


class SuppressOutput:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")

        # Suppress streams
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args, **kwargs):
        # Restore streams
        if self.suppress_stdout:
            sys.stdout = self.original_stdout

        if self.suppress_stderr:
            sys.stderr = self.original_stderr
