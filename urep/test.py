"""Tools for testing. Module can be used instead of unittest."""
import unittest

import numpy as np


class TestCase(unittest.TestCase):
    """Test class that replaces unittest.TestCase."""
    def assertEqualArrays(self, first, second, msg=None):
        """Assert two arrays are equal."""
        first = np.array(first)
        second = np.array(second)
        self.assertEqual(first.tolist(), second.tolist(), msg=msg)

    def assertAlmostEqualArrays(self, first, second,
                                rtol=1e-05, atol=1e-08, equal_nan=False,
                                msg=None):
        """Assert two arrays are almost equal."""
        if np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=equal_nan):
            return
        standard_msg = '{} != {}'.format(repr(first), repr(second))
        msg = self._formatMessage(msg, standard_msg)
        raise self.failureException(msg)


def main():
    unittest.main()
