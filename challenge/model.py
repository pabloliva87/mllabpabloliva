import numpy
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from challenge.utils import get_min_diff
from typing import Tuple, Union, List

class DelayModel:
    Columns_To_Shuffle = ['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']
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
        learning_rate = 0.01
        random_state = 1
        scale_pos_weight = 4.4
        self._model = xgb.XGBClassifier(random_state=random_state,
                                        learning_rate=learning_rate,
                                        scale_pos_weight=scale_pos_weight)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ):# -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
        # TODO: this union causes issues for the interpreter
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
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        data['delay'] = numpy.where(data['min_diff'] > DelayModel.Delay_Threshold, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
        )
        reduced_features = features[DelayModel.Top_10_Features]
        if target_column:
            target = pd.DataFrame(data['delay'])
            return reduced_features, target
        return reduced_features

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
        normal_amount = len(y_train[y_train == 0])
        delayed_amount = len(y_train[y_train == 1])
        scale = normal_amount / delayed_amount
        # TODO: find why scale is being calculated as 1, we'll leave a default initialization for now
        # self._model.scale_pos_weight = scale
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
        prediction = self._model.predict(features)
        result = [int(x) for x in prediction]

        return result
