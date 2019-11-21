from collections import OrderedDict

from .io import read_json


def prepare_config(config, default_config):
    """Set defaults and check fields."""
    if config is None:
        return default_config
    if isinstance(config, str):
        config = read_json(config)
    for key in config:
        if key not in default_config:
            raise ValueError("Unknown parameter {}".format(key))
    new_config = OrderedDict()
    for key, value in default_config.items():
        new_config[key] = config.get(key, value)
    return new_config
