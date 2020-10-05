import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


COL_TIME = 0
COL_PRICE = 1
COL_QUANTITY = 2


def prepare_single_data_window(trades_numpy, window_start, window_end, eval_window_width, req_target_factor, req_stop_factor):
    target_price = trades_numpy[window_end-1, COL_PRICE] * req_target_factor
    stop_price = trades_numpy[window_end-1, COL_PRICE] * req_stop_factor

    eval_window_prices = trades_numpy[window_end:window_end +
                                      eval_window_width, COL_PRICE]

    hit_target_indices = np.where(eval_window_prices >= target_price)[0]
    hit_stop_indices = np.where(eval_window_prices <= stop_price)[0]

    if hit_target_indices.shape[0] == 0:
        # didn't hit target
        y = False
    elif hit_stop_indices.shape[0] == 0:
        # hit target (elsif!) and not stopped out -> success
        y = True
    else:
        # we hit target before stop loss
        y = hit_target_indices[0] < hit_stop_indices[0]

    x = trades_numpy[window_start:window_end].flatten()

    return x, y


def prepare_dataset(trades_numpy, sample_window_width, eval_window_width, req_target_factor=1.005, req_stop_factor=0.999):

    x_list = []
    y_list = []

    for i in range(0, trades_numpy.shape[0] - sample_window_width - eval_window_width + 1, sample_window_width):
        x, y = prepare_single_data_window(
            trades_numpy, i, i + sample_window_width, eval_window_width, req_target_factor, req_stop_factor)

        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


def balance_dataset(X, y, neg_to_pos_ratio=1):
    positive_samples = X[y == True]
    negative_samples = X[y == False]

    pos_count = positive_samples.shape[0]

    if negative_samples.shape[0] > pos_count*neg_to_pos_ratio:
        negative_samples = negative_samples[:pos_count*neg_to_pos_ratio]

    X = np.concatenate((positive_samples, negative_samples))
    y = np.array([True] * positive_samples.shape[0] +
                 [False] * negative_samples.shape[0])

    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    real_true_cnt, real_false_cnt = y_test[y_test ==
                                           True].shape[0], y_test[y_test == False].shape[0]

    pred_true_cnt, pred_false_cnt = y_pred[y_pred ==
                                           True].shape[0], y_pred[y_pred == False].shape[0]

    print(f"Real vs predicted True: {real_true_cnt} vs {pred_true_cnt}")
    print(f"Real vs predicted False: {real_false_cnt} vs {pred_false_cnt}")

    print(metrics.classification_report(y_test, y_pred))


if __name__ == '__main__':

    train_df = pd.read_csv(
        'data/websocket/trades.csv')

    pivot = 70000

    train_data = train_df[:pivot].to_numpy()
    test_data = train_df[pivot:].to_numpy()

    X_train, y_train = prepare_dataset(train_data, 200, 18000)
    X_test, y_test = prepare_dataset(test_data, 200, 18000)

    X_train, y_train = balance_dataset(X_train, y_train, 4)
    X_test, y_test = balance_dataset(X_test, y_test, 4)

    model = train_model(X_train, y_train)

    test_model(model, X_test, y_test)
