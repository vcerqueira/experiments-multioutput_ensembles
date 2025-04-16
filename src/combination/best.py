from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


class BestSingle:

    def __init__(self):
        self.best_model = None

    def fit(self, Y_hat: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        rmse_score = pd.Series({k: root_mean_squared_error(y, Y_hat[k])
                                for k in Y_hat})

        self.best_model = pd.Series(rmse_score).sort_values().index[0]

    def get_weights(self, Y_hat):
        assert self.best_model in Y_hat.columns, 'best model not in preds'

        weights = np.zeros_like(Y_hat)
        weights = pd.DataFrame(weights, columns=Y_hat.columns)
        weights[self.best_model] = 1

        return weights.values
