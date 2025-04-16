import warnings

import pandas as pd

from src.misc.monte_carlo import MonteCarloCV
from src.misc.tde import UnivariateTDE
from src.misc.error import multistep_mae
from src.ensemble.base import MultiOutputEnsemble
from src.ensemble.horizon_weighting import Weighting

warnings.simplefilter('ignore', UserWarning)
CV_SPLIT_TRAIN_SIZE, CV_SPLIT_TEST_SIZE = 0.6, 0.1
CV_N_SPLITS = 10
APPLY_DIFF = True


def cross_val_workflow(series, k, h):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if APPLY_DIFF:
        series = series.diff()

    df = UnivariateTDE(series, k=k, horizon=h)

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]

    mc = MonteCarloCV(n_splits=CV_N_SPLITS,
                      train_size=CV_SPLIT_TRAIN_SIZE,
                      test_size=CV_SPLIT_TEST_SIZE,
                      gap=h + k)

    err_list = []
    for tr_idx, ts_idx in mc.split(X, Y):
        X_tr = X.iloc[tr_idx, :]
        Y_tr = Y.iloc[tr_idx, :]
        X_ts = X.iloc[ts_idx, :]
        Y_ts = Y.iloc[ts_idx, :]
        X_tr_ = X_tr.head(-k - h)
        Y_tr_ = Y_tr.head(-k - h)

        print('Running inner pipeline')
        err = cval_cycle(X_tr_, Y_tr_, X_ts, Y_ts)
        err_list.append(err)

    return err_list


def cval_cycle(X_tr, Y_tr, X_ts, Y_ts):
    """
    :param X_tr:
    :param Y_tr:
    :param X_ts:
    :param Y_ts:
    :return:
    """

    base = MultiOutputEnsemble()
    base.fit_and_trim(X_tr, Y_tr)

    Y_hat_tr = base.predict_all(X_tr)
    Y_hat_tr_h = base.get_yhat_by_horizon(Y_hat_tr)
    Y_hat = base.predict_all(X_ts)
    Y_hat_h = base.get_yhat_by_horizon(Y_hat)

    weights = Weighting(Y_hat_h=Y_hat_h,
                        Y_hat_h_insample=Y_hat_tr_h,
                        Y=Y_ts,
                        Y_insample=Y_tr,
                        X=X_ts,
                        X_insample=X_tr,
                        lambda_=50,
                        omega=0.5)

    Y_hat_f = weights.propagate_from_t1()
    Y_hat_f2 = weights.complete_fh()
    Y_hat_f3 = weights.direct_fh()
    Y_hat_f4 = weights.propagate_from_lt()

    yhf = {**Y_hat_f, **Y_hat_f2, **Y_hat_f3, **Y_hat_f4}

    err = multistep_mae(yhf, Y_ts)

    return err
