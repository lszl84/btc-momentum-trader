import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime

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


def unix_ms_to_datetime_string(unix_ms):
    dispt = datetime.datetime.fromtimestamp(unix_ms/1000.0)
    return str(dispt)


def test_by_simulating_trade(model, trades_numpy, starting_capital, absolute_risk_per_trade, sample_window_width,  req_target_factor=1.005, req_stop_factor=0.999):

    fee = 0.00075
    capital = starting_capital

    all_samples = [trades_numpy[i:i+sample_window_width].flatten()
                   for i in range(0, len(trades_numpy) - sample_window_width)]
    all_preds = model.predict(all_samples)

    i = 0
    while i < len(trades_numpy) - sample_window_width:
        pred = all_preds[i]

        if pred > 0:
            market_index = i + sample_window_width
            buy_price = trades_numpy[market_index, COL_PRICE]
            stop_price = buy_price * req_stop_factor
            target_price = buy_price * req_target_factor

            position_size = absolute_risk_per_trade / (buy_price - stop_price)

            time_str = unix_ms_to_datetime_string(
                trades_numpy[market_index, COL_TIME])

            print(f"[{time_str}]* -OPENED- long position (${buy_price:.3f}), tgt ${target_price:.3f}, stp ${stop_price:.3f}, pos size = {position_size:.6f}")

            # simulate hitting target or stopping out
            i = market_index + 1
            position_closed = False
            while i < len(trades_numpy) and not position_closed:
                current_price = trades_numpy[i, COL_PRICE]

                if current_price <= stop_price:
                    sell_price = current_price
                    position_closed = True
                    close_reason = "STOPPED OUT"

                elif current_price >= target_price:
                    sell_price = target_price
                    position_closed = True
                    close_reason = "TARGET HIT"

                if position_closed:
                    result = position_size * \
                        ((1.0 - fee) * sell_price - (1.0 + fee) * buy_price)

                    capital += result
                    time_str = unix_ms_to_datetime_string(
                        trades_numpy[i, COL_TIME])

                    print(
                        f'[{time_str}] * {close_reason} long position (${buy_price:.3f}) at ${sell_price:.3f} for PnL={result:.3f}. Capital = ${capital:.3f}')

                i += 1
        else:
            i += 1


if __name__ == '__main__':

    train_df = pd.read_csv(
        'data/websocket/trades.csv')

    pivot = 50000
    sample_window_width = 300

    train_data = train_df[:pivot].to_numpy()
    test_data = train_df[pivot:].to_numpy()

    X_train, y_train = prepare_dataset(train_data, sample_window_width, 18000)
    X_test, y_test = prepare_dataset(test_data, sample_window_width, 18000)

    model = train_model(X_train, y_train)

    test_model(model, X_test, y_test)
    test_by_simulating_trade(model, test_data, 600, 0.50, sample_window_width)
