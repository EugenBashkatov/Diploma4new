from scipy.stats import logistic
from scipy.stats import norm
from scipy.stats import pareto
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

def build_pareto(b,par_size):

    data_list = list(pareto.rvs(b, size = par_size))

    return data_list



def logistic_distribution(par_size,par_scale, ploting = False):
    logis = logistic.rvs(size=par_size, scale = par_scale)
    x_logistic = []
    logistic_data = []
    for i in range(0,len(logis)):
        x_logistic.append(i)
        logistic_data.append([i,logis[i]])
    print(logistic_data)
    if ploting == True:
        plt.plot(x_logistic,logis)
        plt.show()
    return logistic_data

def normal_distribution(par_size,par_scale, ploting = False):
    normal = norm.rvs(size=par_size, scale = par_scale)
    x_normal = []
    normal_data = []
    for i in range(0,len(normal)):
        x_normal.append(i)
        normal_data.append([i,normal[i]])
    # print(normal_data)
    # plt.bar(x_normal, normal, align='center')
    if ploting == True:
        plt.plot(x_normal,normal)
        plt.show()
    return normal_data

def create_dis_csv(data):
    with open("output_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


# create_dis_csv(normal_distribution(31,14))


