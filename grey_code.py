from scipy.stats import logistic
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import distributions

def change_input_data(switch,par_size = 10, par_scale = 5):
    if switch == 'normal':
        data = distributions.normal_distribution(par_size,par_scale)
        max_dim = len(data)
    if switch == 'logistic':
        data = distributions.logistic_distribution(par_size, par_scale)
        max_dim = len(data)
    if switch == "daily-test":
        df = pd.read_csv("daily-min-temperatures-01.csv",
                         names=['Date', 'MinTemp'])
        data = df.to_numpy()
        max_dim = sum(1 for my_line in open("daily-min-temperatures-01.csv", 'r'))
    if switch == "daily-10":
        df = pd.read_csv("daily-min-temperatures-02.csv",
                         names=['Date', 'MinTemp'])
        data = df.to_numpy()
        max_dim = sum(1 for my_line in open("daily-min-temperatures-02.csv", 'r'))
    if switch == "daily-30":
        df = pd.read_csv("daily-min-temperatures-03.csv",
                         names=['Date', 'MinTemp'])
        data = df.to_numpy()
        max_dim = sum(1 for my_line in open("daily-min-temperatures-03.csv", 'r'))

    # max_dim = len(data_list)
    return data, max_dim

change_input_data("daily-10")


