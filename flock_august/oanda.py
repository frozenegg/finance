from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import datetime
import json
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np

accountID = '101-009-15953688-001'
access_token = '0c303e5e25bfb774593f5d518c371cb8-6669f843d6d13405a6d7c72be22e49ed'

api = API(access_token=access_token, environment="practice")

def differencial_comparison(days, mean_window):
    currency_pairs = ['USD_JPY', 'EUR_JPY', 'AUD_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY', 'ZAR_JPY']
    s2 = pd.DataFrame()
    for currency_pair in currency_pairs:
        params = {
          "count": days,
          "granularity": "H4"
        }
        r = instruments.InstrumentsCandles(instrument=currency_pair, params=params)
        data = []
        dates = []
        for i in api.request(r)['candles']:
            data.append(i['mid']['o'])
            dates.append(i['time'][:10])

        df = pd.DataFrame(data, columns=['values'])
        df=df.astype(float)

        s = pd.Series(data)
        mean = pd.DataFrame(s.rolling(window=mean_window).mean(), columns=['mean'])
        df1 = pd.concat([df, mean], axis=1)

        s1 = (df1['values'] - df1['mean']) / df1['mean']
        s1 = pd.DataFrame(s1, columns=[currency_pair])
        s2 = pd.concat([s2, s1], axis=1)

    s2.plot()
    print(s2.tail(5))

# differencial_comparison(200, 20)

def differencial_comparison_convolution(days, mean_window):
    currency_pairs = ['USD_JPY', 'EUR_JPY', 'AUD_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY', 'ZAR_JPY']
    s2 = pd.DataFrame()
    for currency_pair in currency_pairs:
        params = {
          "count": days,
          "granularity": "H4"
        }
        r = instruments.InstrumentsCandles(instrument=currency_pair, params=params)
        data = []
        dates = []
        for i in api.request(r)['candles']:
            data.append(i['mid']['o'])
            dates.append(i['time'][:10])

        df = pd.DataFrame(data, columns=['values'])
        df=df.astype(float)

        s = pd.Series(data)
        mean = pd.DataFrame(s.rolling(window=mean_window).mean(), columns=['mean'])
        df1 = pd.concat([df, mean], axis=1)

        s1 = (df1['values'] - df1['mean']) / df1['mean']
        s1 = pd.DataFrame(s1, columns=[currency_pair])
        overall_mean = df1['mean'].mean()
        s1.loc[mean_window - 2, currency_pair] = float(data[mean_window - 1]) / overall_mean
        s1 = s1.cumsum()
        max_value = s1[currency_pair].max(axis=0)
        min_value = s1[currency_pair].min(axis=0)
        max_absolute = max(max_value - 1, abs(min_value - 1))
#         s1 = s1 / max_absolute
        s1 = s1.astype(float)
        s2 = pd.concat([s2, s1], axis=1)

    s2.plot()
    print(s2.tail(30))

# differencial_comparison_convolution(1000, 30)

def differencial_comparison_convolution2(days, mean_window):
    currency_pairs = ['USD_JPY', 'EUR_JPY', 'AUD_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY', 'ZAR_JPY']
    s2 = pd.DataFrame()
    for currency_pair in currency_pairs:
        params = {
          "count": days,
          "granularity": "H4"
        }
        r = instruments.InstrumentsCandles(instrument=currency_pair, params=params)
        data = []
        dates = []
        for i in api.request(r)['candles']:
            data.append(i['mid']['o'])
            dates.append(i['time'][:10])

        df = pd.DataFrame(data, columns=['values'])
        df=df.astype(float)

        s = pd.Series(data)
        mean = pd.DataFrame(s.rolling(window=mean_window).mean(), columns=['mean'])
        df1 = pd.concat([df, mean], axis=1)

        s1 = (df1['values'] - df1['mean']) / df1['mean']
        s1 = pd.DataFrame(s1, columns=[currency_pair])
        overall_mean = df1['mean'].mean()
        s1.loc[mean_window - 2, currency_pair] = float(data[mean_window - 1]) / overall_mean
        s1 = s1.cumsum()
        max_value = s1[currency_pair].max(axis=0)
        min_value = s1[currency_pair].min(axis=0)
        max_absolute = max(max_value - 1, abs(min_value - 1))
        s1 = s1 / max_absolute
        s1 = s1.astype(float)
        s2 = pd.concat([s2, s1], axis=1)

    s2.plot()
    print(s2.tail(30))

# differencial_comparison_convolution(1000, 30)

from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=65)

def coefficiency(counts, mean_window, chosen_pair):
    currency_pairs = ['USD_JPY', 'EUR_JPY', 'AUD_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY', 'ZAR_JPY', 'EUR_USD', 'GBP_USD', 'NZD_USD', 'AUD_USD', 'USD_CHF', 'EUR_CHF', 'GBP_CHF', 'EUR_GBP', 'AUD_NZD', 'AUD_CAD', 'AUD_CHF', 'CAD_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_DKK', 'EUR_NOK', 'EUR_NZD', 'EUR_SEK', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'NZD_CAD', 'NZD_CHF', 'USD_CAD', 'USD_DKK', 'USD_NOK', 'USD_SEK', 'AUD_HKD', 'AUD_SGD', 'CAD_HKD', 'CAD_SGD', 'CHF_HKD', 'CHF_ZAR', 'EUR_CZK', 'EUR_HKD', 'EUR_HUF', 'EUR_PLN', 'EUR_SGD', 'EUR_TRY', 'EUR_ZAR', 'GBP_HKD', 'GBP_PLN', 'GBP_SGD', 'GBP_ZAR', 'HKD_JPY', 'NZD_HKD', 'NZD_SGD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USD_CNH', 'USD_CZK', 'USD_HKD', 'USD_PLN', 'USD_SGD', 'USD_TRY', 'USD_ZAR']
    s2 = pd.DataFrame()

    for currency_pair in currency_pairs:
        params = {
            "count": counts,
            "granularity": "H4"
        }
        r = instruments.InstrumentsCandles(instrument=currency_pair, params=params)
        data = []
        dates = []
        for i in api.request(r)['candles']:
            data.append(i['mid']['o'])
            dates.append(i['time'][:10])

        df = pd.DataFrame(data, columns=['values'])
        df = df.astype(float)

        s = pd.Series(data)
        mean = pd.DataFrame(s.rolling(window=mean_window).mean(), columns=['mean'])
        df1 = pd.concat([df, mean], axis=1)

        s1 = (df1['values'] - df1['mean']) / df1['mean']
        s1 = pd.DataFrame(s1, columns=[currency_pair])
        overall_mean = df1['mean'].mean()
        s1.loc[mean_window - 2, currency_pair] = float(data[mean_window - 1]) / overall_mean
        s1 = s1.cumsum()
        max_value = s1[currency_pair].max(axis=0)
        min_value = s1[currency_pair].min(axis=0)
        max_absolute = max(max_value - 1, abs(min_value - 1))
        s1 = s1 / max_absolute
        s1 = s1.astype(float)
        s2 = pd.concat([s2, s1], axis=1)

    impaired_currency_pairs = currency_pairs
    impaired_currency_pairs.remove(chosen_pair)

    currency_without_one = impaired_currency_pairs[1:]
    X1 = np.array(s2[impaired_currency_pairs[0]]).reshape(-1,1)

    for currency in currency_without_one:
        X = np.array(s2[currency]).reshape(-1,1)
        X1 = np.concatenate([X1, X], axis=1)

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X1[mean_window:])
    y_train_scaled = y_scaler.fit_transform(np.array(s2[chosen_pair])[mean_window:].reshape(-1,1))
    pls.fit(X_train_scaled, y_train_scaled)

    coef = pls.coef_.reshape(1,-1)[0]
    x = np.array(impaired_currency_pairs)
    return x, coef

# x, coef = coefficiency(3000, 20, 'USD_JPY')
# plt.bar(x, coef)

def coefficiency_matrix(counts, mean_window):
    currency_pairs_list = ['USD_JPY', 'EUR_JPY', 'AUD_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY', 'ZAR_JPY', 'EUR_USD', 'GBP_USD', 'NZD_USD', 'AUD_USD', 'USD_CHF', 'EUR_CHF', 'GBP_CHF', 'EUR_GBP', 'AUD_NZD', 'AUD_CAD', 'AUD_CHF', 'CAD_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_DKK', 'EUR_NOK', 'EUR_NZD', 'EUR_SEK', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'NZD_CAD', 'NZD_CHF', 'USD_CAD', 'USD_DKK', 'USD_NOK', 'USD_SEK', 'AUD_HKD', 'AUD_SGD', 'CAD_HKD', 'CAD_SGD', 'CHF_HKD', 'CHF_ZAR', 'EUR_CZK', 'EUR_HKD', 'EUR_HUF', 'EUR_PLN', 'EUR_SGD', 'EUR_TRY', 'EUR_ZAR', 'GBP_HKD', 'GBP_PLN', 'GBP_SGD', 'GBP_ZAR', 'HKD_JPY', 'NZD_HKD', 'NZD_SGD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USD_CNH', 'USD_CZK', 'USD_HKD', 'USD_PLN', 'USD_SGD', 'USD_TRY', 'USD_ZAR']

    whole_data = pd.DataFrame()

    for i in currency_pairs_list:
        print('currently: ' + i + ' (' + str(currency_pairs_list.index(i) + 1) + '/' + str(len(currency_pairs_list)) + ')')
        column, data_array = coefficiency(counts, mean_window, i)
#         print(column, data_array)
        c1 = pd.DataFrame(data=[data_array], columns=column, index=[i])
        c1 = c1.round(5)
#         print(c1.head())
        whole_data = pd.concat([whole_data, c1], axis=0)

#     return whole_data
    whole_data.to_csv('coefficiency_data-' + str(counts) + '-' + str(mean_window) + '.csv')

# column, data_array = coefficiency(3000, 20, 'EUR_JPY')
# plt.bar(column, data_array)

def high_coefficiency_pairs(threshold):
    df = pd.read_csv('coefficiency_data.csv')
    df = df.apply(pd.to_numeric, errors='coerce')
    currency_pairs_list = df.columns

    close_currency_groups = []

    for index, row in df.iterrows():
#         print(currency_pairs_list[index])
        close_currency = []
        for currency_pair in currency_pairs_list:
            if(abs(row[currency_pair]) > threshold):
#                 print(row[currency_pair])
                close_currency.append(currency_pair)
        close_currency_groups.append(close_currency)

    return close_currency_groups

def compair_charts(currency_pair):
    currency_pairs_list = ['USD_JPY', 'EUR_JPY', 'AUD_JPY', 'GBP_JPY', 'NZD_JPY', 'CAD_JPY', 'CHF_JPY', 'ZAR_JPY', 'EUR_USD', 'GBP_USD', 'NZD_USD', 'AUD_USD', 'USD_CHF', 'EUR_CHF', 'GBP_CHF', 'EUR_GBP', 'AUD_NZD', 'AUD_CAD', 'AUD_CHF', 'CAD_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_DKK', 'EUR_NOK', 'EUR_NZD', 'EUR_SEK', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'NZD_CAD', 'NZD_CHF', 'USD_CAD', 'USD_DKK', 'USD_NOK', 'USD_SEK', 'AUD_HKD', 'AUD_SGD', 'CAD_HKD', 'CAD_SGD', 'CHF_HKD', 'CHF_ZAR', 'EUR_CZK', 'EUR_HKD', 'EUR_HUF', 'EUR_PLN', 'EUR_SGD', 'EUR_TRY', 'EUR_ZAR', 'GBP_HKD', 'GBP_PLN', 'GBP_SGD', 'GBP_ZAR', 'HKD_JPY', 'NZD_HKD', 'NZD_SGD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USD_CNH', 'USD_CZK', 'USD_HKD', 'USD_PLN', 'USD_SGD', 'USD_TRY', 'USD_ZAR']
    currency_list_index = currency_pairs_list.index(currency_pair)
    chosen_currency_pairs = high_coefficiency_pairs(0.66)[currency_list_index]

    params = {
            "count": 3000,
            "granularity": "D"
        }
    chosen_currency_pairs.append(currency_pair)

    for i in chosen_currency_pairs:
        r = instruments.InstrumentsCandles(instrument=i, params=params)
        data = []
        for i in api.request(r)['candles']:
            data.append(i['mid']['o'])

        df = pd.DataFrame(data)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.plot()
