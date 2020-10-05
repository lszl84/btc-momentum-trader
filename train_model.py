import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


COL_TIME = 0
COL_PRICE = 1
COL_QUANTITY = 2


def prepare_single_data_window(trades_numpy, window_start, window_end, eval_window_width, req_target_factor=1.005, req_stop_factor=0.999):
    target_price = trades_numpy[window_end-1, COL_PRICE] * req_target_factor
    stop_price = trades_numpy[window_end-1, COL_PRICE] * req_stop_factor

    eval_window_prices = trades_numpy[window_end:window_end +
                                      eval_window_width, COL_PRICE]

    hit_target_indices = np.where(eval_window_prices >= target_price)[0]
    hit_stop_indices = np.where(eval_window_prices <= stop_price)[0]

    if hit_target_indices.shape[0] == 0 or hit_stop_indices.shape[0] == 0:
        # successful trade needs to hit the target, to make sure we earn more than paid with fees
        y = False
    else:
        # we hit target before stop loss
        y = hit_target_indices[0] < hit_stop_indices[0]

    x = trades_numpy[window_start:window_end]

    return x, y
