import os
import pandas as pd

from challenge.model import DelayModel


class ModelWrapper:
    DEFAULT_REPO_ROOT = "/home/pablo/Documents/latamLab/mllabpabloliva/"

    __shared_model = None

    def __init__(self):
        if ModelWrapper.__shared_model is None:
            ModelWrapper.initialize_model()

    @staticmethod
    def initialize_model():
        ModelWrapper.__shared_model = DelayModel()
        root_path = os.environ.get("REPO_ROOT", ModelWrapper.DEFAULT_REPO_ROOT)
        data_location = os.path.join(root_path, "data/data.csv")
        assert os.path.exists(data_location)
        data = pd.read_csv(filepath_or_buffer=data_location, low_memory=False)

        features, target = ModelWrapper.__shared_model.preprocess(
            data=data,
            target_column="delay"
        )

        ModelWrapper.__shared_model.fit(
            features=features,
            target=target
        )

    @staticmethod
    def get_model():
        return ModelWrapper.__shared_model
