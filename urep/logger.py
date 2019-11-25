"""Logger class. Import of this module affects logging module."""
import logging

import tqdm


class Logger(logging.Logger):
    def progress(self, iterator=None, total=None):
        return tqdm.tqdm(iterator,
                         total=total,
                         disable=self.level > logging.INFO)


# Small patch for logging for progress bar.
logging.root = Logger(logging.root.name)
logging.progress = logging.getLogger().progress
