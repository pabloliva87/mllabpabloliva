import logging
import numpy
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from challenge.utils import get_dummy_representation, get_min_diff, reduce_features
#from utils import get_dummy_representation, get_min_diff, reduce_features
from typing import Tuple, Union, List

class DelayModel:
    Delay_Threshold = 15
    Top_10_Features = [ "OPERA_Latin American Wings",
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
    Seed = 42
    Percentage_For_Testing = 0.33

    def __init__(
        self
    ):
        random_state = 1
        learning_rate = 0.01
        scale_pos_weight = 4.4  # Default value, can be overwritten by fit
        self._model = xgb.XGBClassifier(random_state=random_state,
                                        learning_rate=learning_rate,
                                        scale_pos_weight=scale_pos_weight)
        logging.info("Set up model with seed %d, learning_rate %f, scale %f",
                     random_state, learning_rate, scale_pos_weight)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        data['delay'] = numpy.where(data['min_diff'] > DelayModel.Delay_Threshold, 1, 0)
        logging.info("Processing data with %d rows, %d columns",
                     len(data), len(data.columns))

        features = get_dummy_representation(data)
        features = reduce_features(features, DelayModel.Top_10_Features)
        if target_column:
            target = pd.DataFrame(data['delay'])
            return features, target
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                            test_size=DelayModel.Percentage_For_Testing,
                                                            random_state=DelayModel.Seed)
        logging.info("Split data into %d training entries, %d testing entries",
                     len(y_train), len(y_test))
        normal_amount = len(target[target.delay == 0])
        delayed_amount = len(target[target.delay == 1])
        scale = normal_amount / delayed_amount
        logging.debug("Amount of regular flights: %d, delayed flights: %d; calculated scale %f",
                      normal_amount, delayed_amount, scale)
        if scale > 1:
            self._model.scale_pos_weight = scale
        self._model.fit(x_train, y_train)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        logging.debug("Running prediction for %d entries",
                      len(features))
        prediction = self._model.predict(features)
        result = [int(x) for x in prediction]

        return result
