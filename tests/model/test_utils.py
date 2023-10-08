import unittest

from datetime import datetime
import pandas as pd

from challenge.utils import get_min_diff

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_min_diff(
        self
    ):
        test_raw_data_ok = {'Fecha-I': '2017-01-01 6:30:00', 'Fecha-O': '2017-01-01 9:30:00'}
        min_diff_ok = get_min_diff(test_raw_data_ok)
        self.assertEqual(min_diff_ok, 180.0)

        test_raw_data_equals = {'Fecha-I': '2017-01-01 6:30:00', 'Fecha-O': '2017-01-01 6:30:00'}
        min_diff_equals = get_min_diff(test_raw_data_equals)
        self.assertEqual(min_diff_equals, 0.0)

        test_raw_data_after = {'Fecha-I': '2017-01-01 7:30:00', 'Fecha-O': '2017-01-01 5:15:00'}
        min_diff_after = get_min_diff(test_raw_data_after)
        self.assertEqual(min_diff_after, -135.0)
