
import time
import datetime
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, AroonIndicator, IchimokuIndicator
from ta.volume import AccDistIndexIndicator
from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands
import pandas as pd
from multiprocessing import Pool
from binance_connection import remove_ticker_not_in_binance
from matplotlib import pyplot as plt
import numpy as np
import gc
import os
from scipy.signal import argrelextrema
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import re
from pandas import DataFrame
from dydx_v4_connection import LEVERAGE
from keras.models import load_model
from keras_multi_head import MultiHead
from logger import logger

HYSTORICAL_DATA_ROOT = "file_folder/"
plt.switch_backend('agg')
pd.options.mode.chained_assignment = None  # default='warn'
model = load_model('model/keras_model_BI_43200_GMaxPool2d_MH_64.h5',
               custom_objects={'MultiHead': MultiHead})


def convert_columns_to_numeric(fun_df):
    fun_df = fun_df.dropna(axis=0, how='all')
    fun_df = fun_df.apply(pd.to_numeric, errors='ignore')
    return fun_df


def select_base_columns(fun_df):
    fun_df = fun_df[['timestamp', 'BTC_open', 'BTC_high', 'BTC_low', 'BTC_close', 'BTC_volume']]
    return fun_df


def fill_prices_from_candles(fun_df, ticker='BTC'):
    fun_df[ticker] = fun_df[[f'{ticker}_open', f'{ticker}_close']].mean(axis=1)
    return fun_df


def divide_by_mean_volume(fun_df):
    fun_df['BTC_volume'] = fun_df['BTC_volume'] / fun_df['BTC_volume'].mean()
    return fun_df


def flush_memory_variables(variables):
    del(variables)
    gc.collect()


def parallelize_queries(function, queries, mode='imap'):
    """
    This function parallelize the execution of a function: <function> as input
    and an iterable: <queries>. An optional parameter is taken: <QUERY_RATE_LIMIT>:INTEGER which represents the
    maximum number of request per minute
    :param function:
    :param queries:
    :param QUERY_LATE_LIMIT: A number of requests per minute allowed
    :return: res: list --> The iterable of results of the parallel execution of the function defined in the input
    """

    pool = Pool(os.cpu_count()//2)
    if mode == 'imap':
        results = pool.imap(function, queries, chunksize=10)

    elif mode == 'star':
        results = pool.starmap(function, queries, chunksize=10)

    else:
        results = pool.map(function, queries, chunksize=10)

    pool.close()
    pool.join()
    pool.terminate()
    flush_memory_variables([pool])

    return results


def ta_batch_make(candle, day_conversion_factor=60*24, subset=True, ticker='BTC'):
    if not subset:
        if f'{ticker}_index' not in candle.columns:
            candle = candle[[f'{ticker}_open', f'{ticker}_high', f'{ticker}_low', f'{ticker}_close', f'{ticker}_volume', f'{ticker}', f'{ticker}_index_sum']]
        else:
            candle = candle[[f'{ticker}_open', f'{ticker}_high', f'{ticker}_low', f'{ticker}_close', f'{ticker}_volume', f'{ticker}', f'{ticker}_index']]
        print('GENERATING TA FOR N.RECORDS: ', len(candle))
        candle = add_all_ta_features(candle, open=f"{ticker}_open", high=f"{ticker}_high", low=f"{ticker}_low", close=f"{ticker}_close",
                                 volume=f"{ticker}_volume", fillna=True, colprefix=f"{day_conversion_factor//3}", conversion_interval=day_conversion_factor//3)
        candle = add_all_ta_features(candle, open=f"{ticker}_open", high=f"{ticker}_high", low=f"{ticker}_low", close=f"{ticker}_close",
                                 volume=f"{ticker}_volume", fillna=True, colprefix=f"{5*day_conversion_factor//3}", conversion_interval=5*day_conversion_factor//3)
        candle = add_all_ta_features(candle, open=f"{ticker}_open", high=f"{ticker}_high", low=f"{ticker}_low", close=f"{ticker}_close",
                                 volume=f"{ticker}_volume", fillna=True, colprefix=f"{10*day_conversion_factor//3}", conversion_interval=10*day_conversion_factor//3)
        
    if subset:
        bb_3 = BollingerBands(candle[f'{ticker}_close'], window=3*day_conversion_factor, window_dev=1*day_conversion_factor)
        bb_14 = BollingerBands(candle[f'{ticker}_close'], window=14*day_conversion_factor, window_dev=3*day_conversion_factor)
        bb_28 = BollingerBands(candle[f'{ticker}_close'], window=28*day_conversion_factor, window_dev=6*day_conversion_factor)

        avtr_14 = AverageTrueRange(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=14*day_conversion_factor)
        avtr_3 = AverageTrueRange(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=3*day_conversion_factor)
        avtr_28 = AverageTrueRange(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=28*day_conversion_factor)

        accdist = AccDistIndexIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], candle[f'{ticker}_volume'])

        adx_3 = ADXIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=3*day_conversion_factor)
        adx_14 = ADXIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=14*day_conversion_factor)
        adx_28 = ADXIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=28*day_conversion_factor)

        aroon_3 = AroonIndicator(candle[f'{ticker}_close'], window=3*day_conversion_factor)
        aroon_14 = AroonIndicator(candle[f'{ticker}_close'], window=14*day_conversion_factor)
        aroon_28 = AroonIndicator(candle[f'{ticker}_close'], window=28*day_conversion_factor)

        ich_3 = IchimokuIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], window1=1, window2=3, window3=14)
        ich_14 = IchimokuIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], window1=3, window2=14, window3=28)
        ich_28 = IchimokuIndicator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], window1=14*day_conversion_factor, window2=28, window3=56)

        macd_3 = MACD(candle[f'{ticker}_close'], window_slow=14*day_conversion_factor, window_fast=3, window_sign=3*day_conversion_factor)
        macd_14 = MACD(candle[f'{ticker}_close'], window_slow=28*day_conversion_factor, window_fast=14, window_sign=3*day_conversion_factor)
        macd_28 = MACD(candle[f'{ticker}_close'], window_slow=56*day_conversion_factor, window_fast=28, window_sign=14*day_conversion_factor)

        osc_3 = StochasticOscillator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=3*day_conversion_factor, smooth_window=1*day_conversion_factor)
        osc_14 = StochasticOscillator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=14*day_conversion_factor, smooth_window=3*day_conversion_factor)
        osc_28 = StochasticOscillator(candle[f'{ticker}_high'], candle[f'{ticker}_low'], candle[f'{ticker}_close'], window=28*day_conversion_factor, smooth_window=3*day_conversion_factor)

        rsi_3 = RSIIndicator(candle[f'{ticker}_close'], window=3*day_conversion_factor)
        rsi_14 = RSIIndicator(candle[f'{ticker}_close'], window=14*day_conversion_factor)
        rsi_28 = RSIIndicator(candle[f'{ticker}_close'], window=28*day_conversion_factor)


        candle['macd_3'] = macd_3.macd()
        candle['macd_14'] = macd_14.macd()
        candle['macd_28'] = macd_28.macd()

        candle['osc_3'] = osc_3.stoch()
        candle['osc_14'] = osc_14.stoch()
        candle['osc_28'] = osc_28.stoch()

        candle['adx_3'] = adx_3.adx()
        candle['adx_14'] = adx_14.adx()
        candle['adx_28'] = adx_28.adx()

        candle['accdist'] = accdist.acc_dist_index()

        candle['avtr_3'] = avtr_3.average_true_range()
        candle['avtr_14'] = avtr_14.average_true_range()
        candle['avtr_28'] = avtr_28.average_true_range()

        candle['aroon_3'] = aroon_3.aroon_indicator()
        candle['aroon_14'] = aroon_14.aroon_indicator()
        candle['aroon_28'] = aroon_28.aroon_indicator()

        candle['ich_3'] = ich_3.ichimoku_base_line()
        candle['ich_14'] = ich_14.ichimoku_base_line()
        candle['ich_28'] = ich_28.ichimoku_base_line()

        candle['rsi_14'] = rsi_14.rsi()
        candle['rsi_3'] = rsi_3.rsi()
        candle['rsi_28'] = rsi_28.rsi()

        # candle['bb_3'] = bb_3.bollinger_mavg()
        # candle['bb_3_low'] = bb_3.bollinger_lband_indicator()
        # candle['bb_14_high'] = bb_14.bollinger_hband_indicator()
        # candle['bb_14_low'] = bb_14.bollinger_lband_indicator()
        # candle['bb_28_high'] =bb_28.bollinger_hband_indicator()
        # candle['bb_28_low'] =bb_28.bollinger_lband_indicator()


        print('GENERATED TA FOR N.RECORDS: ', len(candle))
    return candle


def shift_column_compute(mydf, custom_interval=14, ticker='BTC'):

    mydf[f'{ticker}_min'] = mydf.iloc[argrelextrema(mydf[f'{ticker}'].to_numpy(), np.less_equal,
                                      order=custom_interval)[0]][f'{ticker}']
    mydf[f'{ticker}_max'] = mydf.iloc[argrelextrema(mydf[f'{ticker}'].to_numpy(), np.greater_equal,
                                      order=custom_interval)[0]][f'{ticker}']

    conditions = [
        mydf[f'{ticker}_max'].isna() & mydf[f'{ticker}_min'].isna(),
        mydf[f'{ticker}_min'].notna(),
        mydf[f'{ticker}_max'].notna()
    ]
    choices = [0, 1, -1]
    mydf[f'{ticker}_index'] = np.select(conditions, choices, default=0)

    return mydf


def add_previous_1000_ticker_data(ticker, interval, end_time=time.time()):
    # fill to 1000 datapoits of 1 day interval, hence 1000 days time spell are 1000/conversion_interval
    if interval == '1d':
        conversion_interval = 0.2
    elif interval == '12h':
        conversion_interval = 0.5
    elif interval == '6h':
        conversion_interval = 0.25
    elif interval == '1h':
        conversion_interval = 1/24
    elif interval == '15m':
        conversion_interval = 15/60/24
    else:
        conversion_interval = 1/60/24
        # conversion_interval = 1/50

    dfs = []
    # for i in range(0, 100):
    for i in range(0, int(1/conversion_interval)):
        if interval == '1d':
            conversion_interval = 1
        # if interval == '1m':
        #     conversion_interval == 1/60/24
        df = remove_ticker_not_in_binance([ticker], interval=interval, end_time=end_time-i*1000*60*60*24*conversion_interval)
        df = pd.DataFrame.from_records(df[ticker]).reset_index(drop=True).iloc[::-1]
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0, ignore_index=True)

    return dfs.iloc[::-1]


def test_result_w_binance_data(df, m=model, interval="1d", ticker='BTC'):
    
    df = fill_prices_from_candles(df, ticker=ticker)
    df = shift_column_compute(df, 14, ticker=ticker)
    df = ta_batch_make(df, day_conversion_factor=1, ticker=ticker)
    df.drop([f'{ticker}_min', f'{ticker}_max'], axis=1, inplace=True)
    df = df.iloc[60::].fillna(0)
    temp_x, temp_y, btc_price = prepare_keras_data(df, ticker=ticker)

    # profit, b = compute_predictions_by_batch(m, temp_x, temp_y, btc_price, batch_size=13)

    y_pred_full = m.predict(temp_x)
    y_pred_full = pd.DataFrame.from_records(y_pred_full)
    a = pd.concat([btc_price.reset_index(drop=True),
                   pd.DataFrame.from_dict(list(temp_y[-len(btc_price):])).reset_index(drop=True),
                   y_pred_full.iloc[-len(btc_price):].reset_index(drop=True),
                   pd.DataFrame.from_dict(np.gradient(np.array(list(y_pred_full[0]), dtype=float), 1))
                   ], axis=1, ignore_index=True)
    a[5] = (a[3]*a[3].shift(1)<0)&(abs(a[3] / a[3].shift(1)) > 0.01)
    a[6] = a[2]
    a[6][~a[5]] = None
    a = standardise_klines(a, delta=13, index_col_index=len(a.columns)-1)
    profit, b = assess_performance(a)

    b.columns = [f'{ticker}', 'Min/Max obs', 'Index', 'dIndex', 'Trigger', 'Action', 'StrategyPerformance', 'BtcPerformance']
    b = b[[f'{ticker}', 'Min/Max obs', 'Index', 'dIndex', 'Trigger', 'StrategyPerformance', 'BtcPerformance', 'Action']]

    # b.columns = [f'{ticker}', 'Min/Max obs', 'Index', 'Trigger_Theo',
    #              'Trigger_Real', 'Theo_Performance', 'StrategyPerformance'
    #              , 'BtcPerformance']
    # b.columns = b.columns.str.replace("Trigger_Real", "Action")
    # b = b[[f'{ticker}', 'Min/Max obs', 'Index', 'Trigger_Theo'
    #              , 'Theo_Performance', 'StrategyPerformance'
    #              , 'BtcPerformance', 'Action']]

    b['time'] = pd.date_range(end=datetime.datetime.today(), periods=len(b))

    prediction_to_return = float(b['Action'].iloc[b['Action'].last_valid_index()])
    b.set_index(keys=['time'], inplace=True)
    b.plot(subplots=True)
    b['Action'].plot(marker='x')
    logger.info("BACKTESTED PROFIT FOR BINANCE DATA: " + str(profit) +
          f" from {datetime.datetime.today()} - n.steps: {len(b)} - step: {interval}")
    graph_data_path = HYSTORICAL_DATA_ROOT + "plots/"
    os.makedirs(os.path.dirname(graph_data_path), exist_ok=True)
    plt.savefig(graph_data_path + 'keras_model_variables_1d_tdx.png')
    plt.close()

    plt.scatter(b.index, b[ticker], c=b['Index'])
    plt.savefig(graph_data_path + 'btc_price_and_index.png')
    plt.close()
    b[ticker].plot()
    plt.scatter(b.index, b[ticker], c=b['Action'], cmap='viridis')
    plt.colorbar()
    plt.savefig(graph_data_path + 'btc_price_and_trigger.png')
    plt.close()
    b.iloc[-100::][ticker].plot()
    plt.scatter(b.iloc[-100::].index, b.iloc[-100::][ticker], c=b.iloc[-100::]['Action'], cmap='viridis')
    plt.colorbar()
    plt.savefig(graph_data_path + 'btc_price_trigger_last_100.png')

    logger.info("Return BTC performance in last 3 months: " + str(float(b['BtcPerformance'].iloc[-1]) / float(b['BtcPerformance'].iloc[-3*30])) + " or: " +
          str(float(b[ticker].iloc[-1]) / float(b[ticker].iloc[-3*30])))
    logger.info("Return in last 3 months: "+ str(float(b['StrategyPerformance'].iloc[-1]) / float(b['StrategyPerformance'].iloc[-3*30])))

    logger.info("Return BTC in last month: "+ str(float(b[ticker].iloc[-1]) / float(b[ticker].iloc[-1*30])))
    logger.info("Return in last month: "+ str(float(b['StrategyPerformance'].iloc[-1]) / float(b['StrategyPerformance'].iloc[-1*30])))

    logger.info("Return if bought and hold: "+ str(float(b[ticker].tail(1)) / float(b[ticker].head(1))))
    logger.info("Return with strategy :"+ str(profit) + " or: "+  str(float(b['StrategyPerformance'].tail(1)) / float(b['StrategyPerformance'].head(1))))

    return prediction_to_return
    # return float(b['Action'].tail(1)) if not pd.isna(float(b['Action'].tail(1))) else float(b['Action'].iloc[-2])


def iterate_gradient_computation_buy_sell(y_pred, batch_size=13):
    real_triggers = []
    for i in range(1, len(y_pred) - batch_size + 1, 1):
        batch = y_pred[: i + batch_size].reshape(i + batch_size)
        gradients = np.gradient(
            batch, edge_order=1)
        gradients_df = pd.DataFrame.from_dict(gradients)
        gradients_df[1] = (gradients_df[0] * gradients_df[0].shift(1) < 0) & (
                    abs(gradients_df[0] / gradients_df[0].shift(1)) > 0.01)
        gradients_df[2] = pd.Series(batch)
        gradients_df[2][~gradients_df[1]] = None
        real_triggers.extend(
            gradients_df[2].tolist()
        ) if i == 1 else real_triggers.append(float(gradients_df[2].iloc[-2]))

    batch = y_pred.reshape(len(y_pred))
    gradients = np.gradient(
        batch, 1)
    gradients_df = pd.DataFrame.from_dict(gradients)
    gradients_df[1] = (gradients_df[0] * gradients_df[0].shift(1) < 0) & (
            abs(gradients_df[0] / gradients_df[0].shift(1)) > 0.01)
    gradients_df[2] = pd.Series(batch)
    gradients_df[2][~gradients_df[1]] = None

    theo_trigger = pd.DataFrame.from_dict(gradients_df[2].tolist())

    return pd.DataFrame.from_dict(real_triggers), theo_trigger


def compute_predictions_by_batch(m, temp_x, temp_y, btc_price, batch_size=30):
    y_pred_full = m.predict(temp_x)
    real_action_df, theo_action_df = iterate_gradient_computation_buy_sell(y_pred_full, batch_size=batch_size)
    a = pd.concat([btc_price.reset_index(drop=True),
                   pd.DataFrame.from_dict(list(temp_y)).reset_index(drop=True),
                   pd.DataFrame.from_records(y_pred_full),
                   theo_action_df,
                   real_action_df
                   ], axis=1, ignore_index=True)

    a = standardise_klines(a, delta=13, index_col_index=len(a.columns) - 1)
    a = standardise_klines(a, delta=13, index_col_index=len(a.columns) - 2)

    theo_profit, b_theo = assess_performance(a, price_col_index=0, index_col_index=3)
    b_theo.drop(len(b_theo.columns) - 1, axis=1, inplace=True)
    real_profit, b_real = assess_performance(b_theo, price_col_index=0, index_col_index=4)
    return real_profit, b_real


def add_iter_stats_col(a):
    a[7] = a[4]
    for index, row in a.iterrows():
        if pd.isna(row[3]) and pd.notna(row[4]) and index >= 45:
            a.at[index+1, 7] = a.at[a[3].iloc[index - 45:index].last_valid_index(), 3]
    a[8] = a[4]
    a[8][a[3].isna()] = np.nan
    return a


def standardise_klines(a, delta=13, index_col_index=-1):
    from copy import copy
    b = copy(a).reset_index(drop=True)
    scaler = MinMaxScaler(feature_range=(0, 1))

    if delta == len(b):
        scaled = scaler.fit_transform(a.iloc[:, index_col_index].to_numpy().reshape(-1, 1))
        scaled_index = pd.DataFrame.from_dict(scaled).reset_index(drop=True)
        b.iloc[:, index_col_index] = scaled_index

    else:
        for i in range(delta, len(a), 1):

            # end_index = i + delta

            if i == delta:
                scaled = scaler.fit_transform(a.iloc[:delta, index_col_index].to_numpy().reshape(-1, 1))
            else:
                scaled = scaler.fit_transform(a.iloc[i-delta:i+1, index_col_index].to_numpy().reshape(-1, 1))

            if i == delta:
                scaled_index = pd.DataFrame.from_dict(scaled).reset_index(drop=True)
                b.iloc[:delta, index_col_index] = scaled_index
            else:
                b.iloc[i, index_col_index] = float(scaled[-1][0])

    return b


def assess_performance(df, initial_capital=1, price_col_index=0, index_col_index=-1,
                       stats=False):
    previous_index = None
    initial_btc_capital = 1
    list_btc_capital = []
    list_capital = []
    previous_btc_price = 0
    wrong_trades = 0
    good_trades = 0
    tot_trades = 0
    good_trades_direction = 0
    good_trades_netfee = 0
    returns = []

    # if is_continuos:
    #     b.iloc[:, index_col_index] = np.where(b.iloc[:, index_col_index] > 0.75, 1, np.where(b.iloc[:, index_col_index] < 0.25, 0, np.nan))

    for i in range(0, len(df), 1):

        # index = df.iloc[i, index_col_index]
        # if abs(previous_index - index) <= threshold and previous_index != 0:
            # df.iloc[i, index_col_index] = None
            # pass
        index = df.iloc[i, index_col_index]
        btc_price = df.iloc[i, price_col_index]

        if not pd.isna(index):

            if (previous_btc_price !=0 and previous_index is not None) and index -previous_index != 0:
                tx_cost = 0.004 * (
                    max([1, abs(index - previous_index)])) * initial_capital
                btc_price = df.iloc[i, price_col_index]
                to_add = previous_index * (
                        (btc_price - previous_btc_price) / previous_btc_price) * initial_capital * LEVERAGE
                change = to_add - tx_cost
                returns.append(change)
                initial_capital += change
                condition = ((btc_price-previous_btc_price)>0 and (index-previous_index) < 0) or (btc_price-previous_btc_price<0 and index-previous_index>0)

                if tx_cost>0:
                    tot_trades += 1
                    if change>=((btc_price-previous_btc_price)/previous_btc_price)*(1-0.004):
                        good_trades += 1
                    else:
                        wrong_trades += 1

                    if condition:
                        good_trades_direction += 1

                    if to_add >=0:
                        good_trades_netfee += 1

                initial_btc_capital += ((
                                                btc_price - previous_btc_price) / previous_btc_price) * initial_btc_capital
            # else:
                # initial_btc_capital += ((btc_price - df.iloc[0, price_col_index])/ df.iloc[0, price_col_index]) * initial_btc_capital

            if previous_index is None or previous_index != index:
                previous_index = index
                previous_btc_price = btc_price

        list_btc_capital.append(initial_btc_capital)
        list_capital.append(initial_capital)

    df_stat = pd.concat([df, pd.DataFrame.from_dict(list_capital), pd.DataFrame.from_dict(list_btc_capital)],
                  ignore_index=True, axis=1)

    if stats:
        logger.info(f"Wrong trades: {wrong_trades}/{tot_trades} - {round(wrong_trades / tot_trades, 3) * 100 if tot_trades>0 else 'N/A'}%")
        logger.info(f"Good trades: {good_trades}/{tot_trades} - {round(good_trades / tot_trades, 3) * 100 if tot_trades>0 else 'N/A'}%")
        logger.info(f"Good trades direction: {good_trades_direction}/{tot_trades} - {round(good_trades_direction / tot_trades, 3) * 100 if tot_trades>0 else 'N/A'}%")
        logger.info(f"Good trades net fees: {good_trades_netfee}/{tot_trades} - {round(good_trades_netfee / tot_trades, 3) * 100 if tot_trades>0 else 'N/A'}%")

        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # assuming daily
        logger.info(f"Sharpe-Ratio: {sharpe}")

    return initial_capital, df_stat


def prepare_keras_data(initial_data, ticker='BTC'):
    initial_data = initial_data[pd.Index(
        [col for col in initial_data.columns if 'unix' in col] + sorted([col for col in initial_data.columns if f'{ticker}_index' not in col and 'unix' not in col
                                                                         and 'volume' not in col.lower()])  +  [col for col in initial_data.columns if
                                                                      f'{ticker}_index' in col])]

    # btc_series_data = initial_data.iloc[:, 1].rolling(2).mean().iloc[1:-1]
    btc_series_data = initial_data['BTC_close'].iloc[2::]
    # btc_series_data = initial_data['BTC_open'].iloc[2::]
    # btc_series_data = initial_data['BTC_close'].iloc[1:-1:]

    # btc_series_data = initial_data.iloc[:, 1]
    initial_data = initial_data.iloc[:-1, :]
    values = initial_data.to_numpy()
    # integer encode direction
    encoder = LabelEncoder()
    values[:, -1] = encoder.fit_transform(values[:, -1])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    previous_steps = 1
    reframed = series_to_supervised(scaled, previous_steps, 1)    # drop columns we don't want to predict
    
    for i in range(0, previous_steps, 1):
        try:
            reframed.drop([f'var1(t{f"-{i}" if i>0 else ""})'], axis=1, inplace=True)
            reframed.drop([f'var29(t-{i+1})'], axis=1, inplace=True)
            #reframed.drop(['var29(t)'], axis=1, inplace=True)
        except:
            pass
    
    reframed = reframed[pd.Index(
        [col for col in reframed.columns if 'var29(t)' not in col] + [col for col in reframed.columns if
                                                                      re.match(r'var29\(t\)', col)])]
    nr_col = -1
    # split into input and outputs
    train_X = reframed.iloc[:, :-1].to_numpy()
    train_y = reframed.iloc[:, -1].to_numpy()

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

    return train_X, train_y, btc_series_data


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
