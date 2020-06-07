from forex_python.converter import CurrencyRates
import datetime
import numpy as np

c = CurrencyRates()
d = datetime.datetime.now()

keys = list(c.get_rates('USD', d).keys())

def all_dates(n):
    day_list = []
    for i in range(n):
        d2 = d + datetime.timedelta(days=-i)
        day_list.append(d2)
    return day_list

def get_rate_data():
    c = CurrencyRates()
    rate_data = []
    day_list = all_dates(33)
    for i in day_list:
        rate = c.get_rates('USD', i)
        rate_data.insert(0, rate)
    return rate_data

def average_20days():
    ave = {}
    rate_data = get_rate_data()
    for key in keys:
        ave[key] = 0
    k = 0
    for i in range(len(rate_data[0])-20, len(rate_data[0])):
        for j in range(len(rate_data)):
            ave[keys[j]] = (ave[keys[j]] * (k) + rate_data[i][keys[j]]) / (k+1)
        k += 1
    return ave

def delta():
    rate_data = get_rate_data()
    ave = average_20days()
    v = []
    for j in keys:
        v_j = []
        for i in range(1, len(rate_data)):
            v_i = rate_data[i][j] - rate_data[i-1][j]
            v_j.append(v_i / ave[j] * 100)
        v.append(v_j)

    return v

def potential():
    pot = []
    rate_data = get_rate_data()
    ave = average_20days()

    for key in keys:
        pot_key = []
        for i in range(len(rate_data)):
            each_key_potential = rate_data[i][key] - ave[key]
            pot_key.append(each_key_potential / ave[key] * 100)
        pot.append(pot_key)
        # print(ave[key])
        # return(pot_key)

    return pot

n = np.random.rand(len(keys)) *2 -1
k = -1
c = 1
v = np.array(delta())
f = k * potential()[0][-1] + c * np.dot(n, v[:,-1])
print(f)
