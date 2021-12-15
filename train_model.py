import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn import metrics
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Flatten

COL_TIME = 0
COL_PRICE = 1
COL_QUANTITY = 2
COL_CNT = 3

TSLA = False

TRAILING_STOP_TRAIN = False
TRAILING_STOP = True
TRAIL_FAC = 2

USE_TREES = True


def prepare_x_window(trades_numpy, window_start, window_end) :
    sl = trades_numpy[window_start:window_end]

    # price_ratio = np.atleast_2d(sl[1:,COL_PRICE]/ sl[:-1,COL_PRICE]).T
    # vol_ratio = np.atleast_2d(sl[1:,COL_QUANTITY]/ sl[:-1,COL_QUANTITY]).T

    # sl =  (sl[1:,1:]/ sl[:-1,1:])**2
    # sl = np.concatenate((sl, price_ratio*vol_ratio),axis=1)

    # sl = price_ratio*vol_ratio
    
    return sl.flatten()

def prepare_single_data_window(trades_numpy, window_start, window_end, eval_window_width, req_target_factor, req_stop_factor):
    entry_price = trades_numpy[window_end-1, COL_PRICE]
    target_price = entry_price * req_target_factor
    stop_price = entry_price * req_stop_factor

    eval_window_prices = trades_numpy[window_end:window_end +
                                      eval_window_width, COL_PRICE]

    if TRAILING_STOP_TRAIN:
        y=None
        stop_size = entry_price - stop_price
        trail_amt = stop_size * TRAIL_FAC
        for p in eval_window_prices:
            if p <= stop_price:
                y=stop_price>entry_price
            elif p >= target_price:
                y=True

            if p - trail_amt > stop_price :
                stop_price = p - trail_amt  
        if y==None: # time exit
            y = eval_window_prices[-1]>entry_price
            
    else:
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
    x = prepare_x_window(trades_numpy, window_start, window_end)

    return x, y


def prepare_dataset(trades_numpy, sample_window_width, eval_window_width, req_target_factor=1.0015, req_stop_factor=0.999):

    x_list = []
    y_list = []

    step = 1#sample_window_width
    tot =trades_numpy.shape[0] - sample_window_width - eval_window_width + 1

    for i in range(0, tot, step):
        print(f"Preparing dataset: {i/tot*100.0:.4f}%...     \r", end='', flush=True)
        x, y = prepare_single_data_window(
            trades_numpy, i, i + sample_window_width, eval_window_width, req_target_factor, req_stop_factor)

        x_list.append(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


def balance_dataset(X, y, neg_to_pos_ratio=1):
    positive_samples = X[y == True]
    negative_samples = X[y == False]

    pos_count = positive_samples.shape[0]
    
    # np.random.shuffle(negative_samples)

    if negative_samples.shape[0] > pos_count*neg_to_pos_ratio:
        negative_samples = negative_samples[:pos_count*neg_to_pos_ratio]

    X = np.concatenate((positive_samples, negative_samples))
    y = np.array([True] * positive_samples.shape[0] +
                 [False] * negative_samples.shape[0])

    return X, y

from numpy import unique
from numpy import random 

def balanced_sample_maker(X, y, random_seed=None):
    """ return a balanced data set by oversampling minority class 
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of positive label
    sample_size = uniq_counts[0]
    over_sample_idx = random.choice(groupby_levels[1], size=sample_size, replace=True).tolist()
    balanced_copy_idx = groupby_levels[0] + over_sample_idx
    random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx, :], y[balanced_copy_idx]


def train_model(X_train, y_train):

    if USE_TREES:
        model = RandomForestClassifier(n_jobs=-1)
        # m_params = { 
        #         "RF": {
        #                 "n_estimators" : np.linspace(2, 500, 500, dtype = "int"),  
        #                 "max_depth": [5, 20, 30, None], 
        #                 "min_samples_split": np.linspace(2, 50, 50, dtype = "int"),  
        #                 "max_features": ["sqrt", "log2",10, 20, None],
        #                 "oob_score": [True],
        #                 "bootstrap": [True]
        #                 },
        #         }
        # scoreFunction = {"recall": "recall", "precision": "precision"}
        # random_search = RandomizedSearchCV(model,
        #                                    n_jobs=-1,
        #                                    param_distributions = m_params["RF"], 
        #                                    n_iter = 20,
        #                                    scoring = scoreFunction,               
        #                                    refit = "precision",
        #                                    return_train_score = True,
        #                                    random_state = 42,
        #                                    cv = 5,
        #                                     verbose = True) 

        # #trains and optimizes the model
        # random_search.fit(X_train, y_train)

        # #recover the best model
        # model = random_search.best_estimator_

        model.fit(X_train, y_train)
    else:
        input_shape = X_train.shape[1]
        num_sensors = COL_CNT
        periods = int(input_shape/num_sensors)
        num_classes = 2

        print(X_train.shape)
        print(input_shape)
        print(y_train.shape)

        cat_y = tf.keras.utils.to_categorical(y_train, num_classes=num_classes, dtype='float32')
        print(cat_y.shape)

        

        model = Sequential()
        model.add(Reshape((periods, num_sensors), input_shape=(input_shape,)))
        model.add(Conv1D(132,4, activation='relu', input_shape=(periods,num_sensors)))
        model.add(Conv1D(132,4, activation='relu'))
        model.add(MaxPooling1D(3))
        # model.add(Conv1D(132,4,activation='relu'))
        # model.add(Conv1D(132,4,activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        # model.add(Flatten())
        model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 512
        epochs =30

        model.fit(X_train, cat_y, batch_size=batch_size, epochs=epochs, validation_split=0.2,verbose=1)

    return model


def test_model(model, X_test, y_test):
    print("Predicting samples...")
    
    if not USE_TREES:
        y_pred = model.predict(X_test, batch_size=512, verbose=1)
        print("Preds ready, converting from categorical...")
        y_pred = np.argmax(y_pred, axis=-1)
    else:
        y_pred = model.predict(X_test)

    real_true_cnt, real_false_cnt = y_test[y_test ==
                                           True].shape[0], y_test[y_test == False].shape[0]

    pred_true_cnt, pred_false_cnt = y_pred[y_pred ==
                                           True].shape[0], y_pred[y_pred == False].shape[0]

    print(f"Real vs predicted True: {real_true_cnt} vs {pred_true_cnt}")
    print(f"Real vs predicted False: {real_false_cnt} vs {pred_false_cnt}")

    print(metrics.classification_report(y_test, y_pred))


def print_ds_stats(y):
    true_samples, false_samples=y[y == True].shape[0],y[y == False].shape[0]
    print(f"True vs false samples: {true_samples} vs {false_samples}. Total = {true_samples+false_samples}")


def unix_ms_to_datetime_string(unix_ms):
    dispt = datetime.datetime.fromtimestamp(unix_ms/1000.0)
    return str(dispt)


def test_by_simulating_trade(model, trades_numpy, starting_capital, absolute_risk_per_trade, sample_window_width, sma, req_target_factor=1.0012, req_stop_factor=0.999):

    fee =0 #0.00075
    fee_per_001 = 0.0035*2
    capital = starting_capital

    if model is not None:
        print("Preparing samples list for prediction...")
        all_samples = [prepare_x_window(trades_numpy, i, i+sample_window_width)#trades_numpy[i:i+sample_window_width].flatten()
                    for i in range(0, len(trades_numpy) - sample_window_width)]
        all_samples = np.stack(all_samples)
        print("Stacking samples to numpy...")
        print(all_samples.shape)
        # exit(0)
        print("Predicting samples...")
        
        if not USE_TREES:
            all_preds = model.predict(all_samples, batch_size=512, verbose=1)
            print("Preds ready, converting from categorical...")
            print(all_preds.shape)
            all_preds = np.argmax(all_preds, axis=-1) # TODO: HEERE
        else:
            all_preds = model.predict(all_samples)
        print("Preds ready.")
        print(all_preds.shape)
        # exit(0)

    i = 0
    R = 0
    trade_cnt = 0
    abfeesum=0
    relfeesum = 0
    total_traded = 0

    sma_idx = 0
    

    while i < len(trades_numpy) - sample_window_width:

        if model is not None:
            pred = all_preds[i]
        else:
            pred = 1

        market_index = i + sample_window_width

        if sma is not None:
            sma_row = sma.iloc[sma_idx]
            # closing timestamp!
            while sma_idx < len(sma) and sma_row["timestamp"]< trades_numpy[market_index, COL_TIME] :
                sma_idx+=1
                sma_row = sma.iloc[sma_idx]
                
            # print(sma_row["timestamp"], trades_numpy[market_index, COL_TIME])

            prev_sma = None
            if sma_idx>0 and not np.isnan(sma.iloc[sma_idx-1]["sma"]):
                prev_sma = sma.iloc[sma_idx-1]["sma"]

            sma_cross = False

            if prev_sma is not None:
                # print(prev_sma, trades_numpy[market_index, COL_PRICE])
                if (trades_numpy[market_index, COL_PRICE] > prev_sma) and trades_numpy[market_index-1, COL_PRICE] < prev_sma:
                    sma_cross = True
        else: 
            sma_cross = True

        # if (pred>0 and not above_sma) :
        #     print("Prediction failed, because not above sma")

        if pred > 0 and sma_cross:
            # if sma is not None and model is not None:

            trade_time=unix_ms_to_datetime_string(trades_numpy[market_index, COL_TIME])
            print(trade_time,"CumR = ", R, "Pred!", trades_numpy[market_index, COL_PRICE],trades_numpy[market_index-1, COL_PRICE])
            
            buy_price = trades_numpy[market_index, COL_PRICE]
            stop_price = buy_price * req_stop_factor
            target_price = buy_price * req_target_factor

            stop_size = buy_price - stop_price
            trail_amt = stop_size * TRAIL_FAC

            position_size = absolute_risk_per_trade / (buy_price - stop_price)
            

            # time_str = unix_ms_to_datetime_string(
            #     trades_numpy[market_index, COL_TIME])

            #print(f"[{time_str}]* -OPENED- long position (${buy_price:.3f}), tgt ${target_price:.3f}, stp ${stop_price:.3f}, pos size = {position_size:.6f}")

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
                    total_traded += position_size
                    trade_cnt +=1
                    absolute_fee = position_size*fee_per_001 if TSLA else position_size*100*fee_per_001
                    result = position_size * \
                        ((1.0 - fee) * sell_price - (1.0 + fee) * buy_price) \
                            - absolute_fee

                    capital += result
                    time_str = unix_ms_to_datetime_string(
                        trades_numpy[i, COL_TIME])

                    #print(f'[{time_str}] * {close_reason} long position (${buy_price:.3f}) at ${sell_price:.3f} for PnL={result:.3f}. Capital = ${capital:.3f}')

                    R += result / absolute_risk_per_trade
                    abfeesum += absolute_fee
                    relfeesum += 0.00075 * position_size * (buy_price + sell_price)

                if TRAILING_STOP and current_price -trail_amt > stop_price:
                    stop_price = current_price - trail_amt  

                i += 1
        else:
            i += 1

    start_time = unix_ms_to_datetime_string(trades_numpy[0, COL_TIME])
    end_time = unix_ms_to_datetime_string(trades_numpy[-1, COL_TIME])

    print(f'From [{start_time}] to [{end_time}]: Trades = {trade_cnt}, R = {R}. Sim absolute fees = ${abfeesum}. Sim rel fees = ${relfeesum}. With rel fees R = {(R*absolute_risk_per_trade+abfeesum-relfeesum)/absolute_risk_per_trade}. Total BTC traded = {total_traded}')


if __name__ == '__main__':

    print(pd.__version__)

    if TSLA:
        train_df = pd.read_csv('tsla.csv')
        pivot = 300
        cutoff=-1
        sample_window_width = 10
        check_width = 50
    else:
        # train_df = pd.read_csv('data/websocket/trades.csv')
        train_df = pd.read_csv('data/websocket/historical-reversed-trades.csv')
        train_df.sort_values(by=['id'], inplace=True)
        train_df.drop('id', axis=1, inplace=True)
        print(train_df)
        pivot =20000000
        cutoff = -1#2*pivot
        sample_window_width = 50
        check_width = 1500

    train_data = train_df[:pivot].to_numpy()
    test_data = train_df[pivot:cutoff].to_numpy()


    # if TSLA:
    #     train_data = train_df[:pivot].to_numpy()
    #     test_data = train_df[pivot:cutoff].to_numpy()
    # else:
    #     train_data = train_df[:pivot].to_numpy()
    #     test_df = pd.read_csv('tsla.csv')
    #     test_data = test_df.to_numpy()

    print("All Samples (train + test):", train_data.shape[0]+test_data.shape[0])

    

    X_train, y_train = prepare_dataset(train_data, sample_window_width, check_width)

    print("Training set before balancing:")
    print_ds_stats(y_train)

    print("Shape:", X_train.shape)

    # X_train, y_train = balanced_sample_maker(X_train, y_train)

    # print("Training set after balancing:")
    # print_ds_stats(y_train)

   

    # X_test, y_test=balance_dataset(X_test, y_test,3)

    # print("Test set after balancing:")
    # print_ds_stats(y_test)


    model = train_model(X_train, y_train)

    X_test, y_test = prepare_dataset(test_data, sample_window_width, check_width)

    print("Test set stats:")
    print_ds_stats(y_test)

    test_model(model, X_test, y_test)



    # BLAH -------------------
    # test_df = train_df[pivot:].copy()
    # if TSLA:
    #     test_df["T"] = pd.to_datetime(test_df["T"], unit="s")
    # else:
    #     test_df["T"] = pd.to_datetime(test_df["T"], unit="ms")

    # print(test_df)
    
    # g=test_df.groupby(pd.Grouper(freq="15Min", key="T"))
    # # sma=df.rolling(window=5).mean()

    

    # g=g.nth([-1])
    # g["sma"]=g["p"].rolling(window=10).mean()


    # # print(g[g["sma"] > g["p"]])

    # if TSLA:
    #     k = 10**9
    # else:
    #     k=10**6

    # g['timestamp'] = g['T'].apply(lambda x: x.value/k).astype(int)

    # print(g) 


    # print("Results by just sma, not model")
    # test_by_simulating_trade(None, test_data, 10000, 100, sample_window_width, g)

    # print("Results by sma and model")
    # test_by_simulating_trade(model, test_data, 10000, 100, sample_window_width, g)


    if USE_TREES:
        import time
        start_time = time.time()
        importances = model.feature_importances_
        std = np.std([
            tree.feature_importances_ for tree in model.estimators_], axis=0)
        elapsed_time = time.time() - start_time

        print(f"Elapsed time to compute the importances: "
            f"{elapsed_time:.3f} seconds")

        forest_importances = pd.Series(importances)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        # plt.show()
        plt.show(block=False)



    print("Results without checking sma")
    test_by_simulating_trade(model, test_data, 10000, 100, sample_window_width, None)
    
    plt.show()