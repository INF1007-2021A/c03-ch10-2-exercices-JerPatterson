#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
from sklearn import model_selection, ensemble, linear_model
import matplotlib.pyplot as plt
import pandas as pd


# TODO: Définissez vos fonctions ici
def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, ";")


def separate_value_wanted(df_all_parameters: pd.DataFrame) -> tuple:
    df_parameter_wanted = df_all_parameters["quality"]
    df_other_parameters = df_all_parameters.drop("quality", axis = 1)

    return df_parameter_wanted, df_other_parameters


def separate_data_in_half(df_wanted: pd.DataFrame, df_other: pd.DataFrame) -> tuple:
    return model_selection.train_test_split(df_wanted, df_other)


def predictions_random_forest(other_values: pd.DataFrame, targeted_values: pd.DataFrame, test_other_values: pd.DataFrame) -> list:
    regression = ensemble.RandomForestRegressor().fit(other_values, targeted_values)
    return regression.predict(test_other_values)


def predictions_linear_regression(other_values: pd.DataFrame, targeted_values: pd.DataFrame, test_other_values: pd.DataFrame) -> list:
    regression = linear_model.LinearRegression().fit(other_values, targeted_values)
    return regression.predict(test_other_values)


def predictions_analysis(model: str, predictions: list, targeted_values: list) -> None:
    index = [number for number in range(len(predictions))]
    plt.plot(index, targeted_values, 'b', label = "Targeted values")
    plt.plot(index, predictions, 'darkorange', label = "Predicted values")
    plt.xlabel('Number of samples')
    plt.ylabel('Quality')
    plt.legend()
    plt.title(f"{model} predictions analysis")
    plt.show()

def mean_squared_error(predictions, targeted_values):
    pass


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    df_white_wines = read_csv_file("data/winequality-white.csv")

    df_quality_wanted, df_other_data = separate_value_wanted(df_white_wines)
    train_quality, test_quality, train_other, test_other = separate_data_in_half(df_quality_wanted, df_other_data)
    random_forest_model = predictions_random_forest(train_other, train_quality, test_other)
    linear_regression_model = predictions_linear_regression(train_other, train_quality, test_other)
    
    predictions_analysis('RandomForestRegressor', random_forest_model, test_quality)
    predictions_analysis('LinearRegression', linear_regression_model, test_quality)
