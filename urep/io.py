"""Tools for simple input and output."""
import json
import os
from collections import OrderedDict


def ensure_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isfile(path):
        raise RuntimeError("{} is a file".format(path))


class ConfigJSONEncoder(json.JSONEncoder):
    """Encoder that dumps arrays to single line."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def iterencode(self, o, **kwargs):
        return self._iterencode(o)

    def _iterencode(self, o):
        indent = " " * self.indent
        if isinstance(o, dict):
            if len(o) == 0:
                yield "{}"
                return
            yield "{\n"
            prev_line = None
            for k, v in o.items():
                if not isinstance(k, str):
                    raise TypeError("JSON keys should be strings")
                k_encoded = indent + "\"{}\": ".format(k)
                for i, v_encoded in enumerate(self._iterencode(v)):
                    if i == 0:
                        if prev_line is not None:
                            yield prev_line + ",\n"
                        prev_line = k_encoded + v_encoded
                    else:
                        yield prev_line
                        prev_line = indent + v_encoded
            yield prev_line + "\n"
            yield "}"
            return
        if self._is_plain_array(o):
            encoded = [v_encoded for v in o for v_encoded in self._iterencode(v)]
            yield "[" + ", ".join(encoded) + "]"
            return
        if isinstance(o, list):
            if len(o) == 0:
                return "[]"
            yield "[\n"
            prev_line = None
            for v in o:
                for i, v_encoded in enumerate(self._iterencode(v)):
                    if i == 0:
                        if prev_line is not None:
                            yield prev_line + ",\n"
                    else:
                        yield prev_line
                    prev_line = indent + v_encoded
            yield prev_line + "\n"
            yield "]"
            return
        for line in super().iterencode(o):
            yield line
        return

    @staticmethod
    def _is_plain_array(o):
        """Check input is list without nested dicts and lists."""
        if not isinstance(o, list):
            return False
        for v in o:
            if isinstance(v, list):
                return False
            if isinstance(v, dict):
                return False
        return True


def read_json(filename):
    """Read json from file."""
    with open(filename) as fp:
        return json.load(fp, object_pairs_hook=OrderedDict)


def write_json(obj, filename):
    """Write json file with custom formating."""
    with open(filename, "w") as fp:
        json.dump(obj, fp, cls=ConfigJSONEncoder, indent=4)
