import torch


class ModelBase(torch.nn.Module):
    @property
    def output_dims(self):
        raise NotImplementedError("Base class method is not implemented")
