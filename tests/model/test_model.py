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
        """
        with open('/home/pablo/Documents/latamLab/mllabpabloliva/report.txt', 'w+') as reportFile:
            reportFile.write(str(report))
            reportFile.write('\n')
            reportFile.write(str(self.model._model))
            reportFile.write('\n')
            reportFile.write(str(self.model._model.scale_pos_weight))
            reportFile.write('\n')
        #print(report)
        #logging.warning("recall " + str(report["0"]["recall"]))
        #logging.warning("f1-score " + str(report["0"]["f1-score"]))
        #logging.warning("recall " + str(report["1"]["recall"]))
        #logging.warning("f1-score " + str(report["1"]["f1-score"]))
        """

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
