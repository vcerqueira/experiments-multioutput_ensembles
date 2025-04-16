from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.misc.normalization import normalize_and_proportion


class LossTrain:

    def __init__(self):
        self.rmse = None
        self.weights = None

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.rmse_ = pd.Series({k: root_mean_squared_error(y, Y_hat[k])
                                for k in Y_hat})

        self.weights = normalize_and_proportion(-self.rmse_)

    def get_weights(self, Y_hat: pd.DataFrame):
        weights = np.array([self.weights, ] * Y_hat.shape[0])

        return weights
