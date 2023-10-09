import unittest
import logging
import os
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
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

    TARGET_COL = [
        "delay"
    ]

    DEFAULT_REPO_ROOT = "/home/pablo/Documents/latamLab/mllabpabloliva/"


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        root_path = os.environ.get("REPO_ROOT", TestModel.DEFAULT_REPO_ROOT)
        data_location = os.path.join(root_path, "data/data.csv")
        assert os.path.exists(data_location)
        self.data = pd.read_csv(filepath_or_buffer=data_location, low_memory=False)

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        training_testing_split = train_test_split(features, target, test_size = 0.33, random_state = 42)
        features_training = training_testing_split[0]
        features_validation = training_testing_split[1]
        target_training = training_testing_split[2]
        target_validation = training_testing_split[3]

        self.model.fit(
            features=features_training,
            target=target_training
        )

        predicted_targets = self.model.predict(
            features=features_validation
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features_validation.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)

    def test_model_constructor(
        self
    ):
        self.assertTrue(self.model is not None)

    def test_validate_input(
        self
    ):
        valid_data_raw = {"OPERA": {0: "Aerolineas Argentinas"},
                          "TIPOVUELO": {0: "N"},
                          "MES": {0: 3}
                          }
        valid_data = pd.DataFrame(valid_data_raw)
        self.assertTrue(self.model.validate_input(valid_data))

        invalid_operator_raw = {"OPERA": {0: ""},
                                "TIPOVUELO": {0: "N"},
                                "MES": {0: 3}
                                }
        invalid_operator_data = pd.DataFrame(invalid_operator_raw)
        self.assertFalse(self.model.validate_input(invalid_operator_data))

        invalid_flight_type_raw = {"OPERA": {0: "LATAM"},
                                   "TIPOVUELO": {0: "O"},
                                   "MES": {0: 3}
                                   }
        invalid_flight_type_data = pd.DataFrame(invalid_flight_type_raw)
        self.assertFalse(self.model.validate_input(invalid_flight_type_data))

        invalid_month_high_raw = {"OPERA": {0: "LATAM"},
                             "TIPOVUELO": {0: "I"},
                             "MES": {0: 14}
                             }
        invalid_month_high_data = pd.DataFrame(invalid_month_high_raw)
        self.assertFalse(self.model.validate_input(invalid_month_high_data))

        invalid_month_low_raw = {"OPERA": {0: "LATAM"},
                             "TIPOVUELO": {0: "I"},
                             "MES": {0: 0}
                             }
        invalid_month_low_data = pd.DataFrame(invalid_month_low_raw)
        self.assertFalse(self.model.validate_input(invalid_month_low_data))
