"""Helper tools."""
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
