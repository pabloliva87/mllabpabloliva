from datetime import datetime
import logging
import pandas as pd


def adjust_dummy_columns(features, intended_columns):
    """ Adds and removes columns to a dataframe so that it fits the prediction model

        Args:
            features (pd.DataFrame): raw data.
            intended_columns (list[str]): the columns that the result dataframe shall have.

        Returns:
            pd.DataFrame: features.
    """
    for column in intended_columns:
        if column not in features:
            features[column] = 0
    features = reduce_features(features, intended_columns)
    return features


def get_dummy_representation(data):
    """ Transforms certain columns of a dataframe into dummy representation

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            pd.DataFrame: features.
    """
    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix='OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
        pd.get_dummies(data['MES'], prefix='MES')],
        axis=1
    )
    return features


def get_min_diff(data):
    """
        Calculates the difference in minutes between the intended departure date and the actual departure date

        Args:
            data (pd.DataFrame): flight data.

        Returns:
            float: the amount of minutes elapsed between the intended departure date
                   and the actual departure date (can be negative).
    """
    leaving_in_advance_tolerance = 3600.0  # seconds, an hour
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    if (fecha_i - fecha_o).total_seconds() > leaving_in_advance_tolerance:
        message = ("Found flight leaving way before departure time: "
                   "flew at {}, intended time {}".format(fecha_o, fecha_i))
        logging.warning(message)
    minutes_difference = ((fecha_o - fecha_i).total_seconds()) / 60.0
    return minutes_difference


def reduce_features(features, columns_to_keep):
    """
        Reduces the dataframe features to the columns in columns_to_keep

        Args:
            features (pd.DataFrame): raw data.
            columns_to_keep (list[str]): the columns that will remain.

        Returns:
            pd.DataFrame: features.
    """
    reduced_features = features[columns_to_keep]
    logging.debug("Reduced features to keep only %s",
                  str(columns_to_keep))

    return reduced_features

