#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
from sklearn import model_selection, ensemble, linear_model
import pandas as pd
from sklearn.base import RegressorMixin


# TODO: Définissez vos fonctions ici
def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, ";")


def separate_value_wanted(df_all_parameters: pd.DataFrame) -> tuple:
    df_parameter_wanted = df_all_parameters["quality"]
    df_other_parameters = df_all_parameters.drop("quality", axis = 1)

    return df_parameter_wanted, df_other_parameters


def separate_data_in_half(df_wanted: pd.DataFrame, df_other: pd.DataFrame) -> tuple:
    return model_selection.train_test_split(df_wanted, df_other)


def train_random_forest(other_values: pd.DataFrame, targeted_values: pd.DataFrame) -> 'ensemble.RandomForestRegressor':
    return ensemble.RandomForestRegressor().fit(other_values, targeted_values)


def evaluate_random_forest(regression: 'ensemble.RandomForestRegressor', test_other_values: pd.DataFrame) -> list:
    return regression.predict(test_other_values)


def train_linear_regression(other_values: pd.DataFrame, targeted_values: pd.DataFrame) -> 'linear_model.LinearRegression':
    return linear_model.LinearRegression().fit(other_values, targeted_values)


def evaluate_linear_regression(regression: 'linear_model.LinearRegression', test_other_values: pd.DataFrame) -> list:
    return regression.predict(test_other_values)


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    df_white_wines = read_csv_file("data/winequality-white.csv")

    df_quality_wanted, df_other_data = separate_value_wanted(df_white_wines)
    train_quality, test_quality, train_other, test_other = separate_data_in_half(df_quality_wanted, df_other_data)

    random_forest_model = evaluate_random_forest(train_random_forest(train_other, train_quality), test_other)
    linear_regression_model = evaluate_linear_regression(train_linear_regression(train_other, train_quality), test_other)
    print(random_forest_model, len(random_forest_model))
    print(linear_regression_model, len(linear_regression_model))