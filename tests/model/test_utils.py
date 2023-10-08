import unittest

from datetime import datetime
import pandas as pd

from challenge.utils import adjust_dummy_columns, get_dummy_representation, get_min_diff, reduce_features

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.raw_data = {'Fecha-I': {0: '2017-01-01 23:30:00',
                                1: '2017-01-02 23:30:00',
                                2: '2017-01-03 23:30:00',
                                3: '2017-01-04 23:30:00',
                                4: '2017-01-05 23:30:00'},
                    'Vlo-I': {0: '226', 1: '226', 2: '226', 3: '226', 4: '226'},
                    'Ori-I': {0: 'SCEL', 1: 'SCEL', 2: 'SCEL', 3: 'SCEL', 4: 'SCEL'},
                    'Des-I': {0: 'KMIA', 1: 'KMIA', 2: 'KMIA', 3: 'KMIA', 4: 'KMIA'},
                    'Emp-I': {0: 'AAL', 1: 'AAL', 2: 'AAL', 3: 'AAL', 4: 'AAL'},
                    'Fecha-O': {0: '2017-01-01 23:33:00',
                                1: '2017-01-02 23:39:00',
                                2: '2017-01-03 23:39:00',
                                3: '2017-01-04 23:33:00',
                                4: '2017-01-05 23:28:00'},
                    'Vlo-O': {0: '226', 1: '226', 2: '226', 3: '226', 4: '226'},
                    'Ori-O': {0: 'SCEL', 1: 'SCEL', 2: 'SCEL', 3: 'SCEL', 4: 'SCEL'},
                    'Des-O': {0: 'KMIA', 1: 'KMIA', 2: 'KMIA', 3: 'KMIA', 4: 'KMIA'},
                    'Emp-O': {0: 'AAL', 1: 'AAL', 2: 'AAL', 3: 'AAL', 4: 'AAL'},
                    'DIA': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
                    'MES': {0: 7, 1: 10, 2: 12, 3: 4, 4: 11},
                    'AÑO': {0: 2017, 1: 2017, 2: 2017, 3: 2017, 4: 2017},
                    'DIANOM': {0: 'Domingo',
                               1: 'Lunes',
                               2: 'Martes',
                               3: 'Miercoles',
                               4: 'Jueves'},
                    'TIPOVUELO': {0: 'I', 1: 'I', 2: 'I', 3: 'N', 4: 'N'},
                    'OPERA': {0: 'American Airlines',
                              1: 'Sky Airline',
                              2: 'Latin American Wings',
                              3: 'Grupo LATAM',
                              4: 'Copa Air'},
                    'SIGLAORI': {0: 'Santiago',
                                 1: 'Santiago',
                                 2: 'Santiago',
                                 3: 'Santiago',
                                 4: 'Santiago'},
                    'SIGLADES': {0: 'Miami', 1: 'Miami', 2: 'Miami', 3: 'Miami', 4: 'Miami'},
                    'period_day': {0: 'noche', 1: 'noche', 2: 'noche', 3: 'noche', 4: 'noche'},
                    'high_season': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                    'min_diff': {0: 3.0, 1: 9.0, 2: 9.0, 3: 3.0, 4: -2.0},
                    'delay': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                    }
        self.data = pd.DataFrame(self.raw_data)
        self.columns_to_keep = [ "OPERA_Latin American Wings",
                                 "MES_7",
                                 "MES_10",
                                 "OPERA_Grupo LATAM",
                                 "MES_12",
                                 "TIPOVUELO_I",
                                 "MES_4",
                                 "MES_11",
                                 "OPERA_Sky Airline",
                                 "OPERA_Copa Air"
                               ]

    def test_adjust_dummy_columns(
        self
    ):
        features_raw = {'MES_8': {0: 0, 1: 1, 2: 0},
                        'MES_7': {0: 1, 1: 0, 2: 0},
                        'MES_6': {0: 0, 1: 0, 2: 1},
                        'OPERA_Aeromexico': {0: 1, 1: 0, 2: 1},
                        'OPERA_Sky Airline': {0: 0, 1: 1, 2: 0},
                        'TIPOVUELO_I': {0: 1, 1: 1, 2: 0},
                       }
        features_pd = pd.DataFrame(features_raw)
        features_adjusted = adjust_dummy_columns(features_pd, self.columns_to_keep)
        for column in self.columns_to_keep:
            self.assertTrue(column in features_adjusted)
        for column_name in features_raw.keys():
            if column_name in self.columns_to_keep:
                self.assertTrue(column_name in features_adjusted)
            else:
                self.assertFalse(column_name in features_adjusted)

    def test_get_dummy_representation(
        self
    ):
        processedData = get_dummy_representation(self.data)
        self.assertEqual(5, len(processedData))
        for column in processedData.columns:
            self.assertTrue('OPERA' in column or 'TIPOVUELO' in column or 'MES' in column)

    def test_reduce_features(
        self
    ):
        features_raw = {'MES_8': {0: 0, 1: 1, 2: 0},
                        'MES_7': {0: 1, 1: 0, 2: 0},
                        'MES_6': {0: 0, 1: 0, 2: 1},
                        'OPERA_Aeromexico': {0: 1, 1: 0, 2: 1},
                        'OPERA_Sky Airline': {0: 0, 1: 1, 2: 0},
                        'TIPOVUELO_I': {0: 1, 1: 1, 2: 0},
                       }
        features_pd = pd.DataFrame(features_raw)
        columns_to_keep = ['MES_6', 'OPERA_Aeromexico']
        features_adjusted = reduce_features(features_pd, columns_to_keep)
        for column_name in features_raw.keys():
            if column_name in columns_to_keep:
                self.assertTrue(column_name in features_adjusted)
            else:
                self.assertFalse(column_name in features_adjusted)


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
