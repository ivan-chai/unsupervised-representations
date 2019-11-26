"""DataParallel wrapper."""
import torch


class DataParallel(torch.nn.DataParallel):
    """Optional data parallel support with synchronized batchnorm."""
    def __init__(self, module):
        super().__init__(module)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)
