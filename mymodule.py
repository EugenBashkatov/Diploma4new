import statistics
import sys
import os
import math
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, norm, poisson, binom, expon
import stem_plot as st
import datetime as dt
import seaborn as sns
import utilities as ut
import distributions as dst
# numm=-math.inf
# if numm<10000000:print (math.atan(numm))
# exit()
import xlsxwriter
NEED_NORMALIZE =True
DEBUG=True
DEBUG1 = False
DEBUG2 = False
DEBUG3 = False
DEBUG4 = False
DEBUG5 = False
def print_hi(name):
    # Use a breakpoint in the code get_k below to debug your script.
    print(f'Hi, {name}')
if __name__ == '__main__':
    print_hi('mymodule')
    d_cluster_chain = {0: [0]}
    d_cluster_length = {0: 2}
    d_cluster_mean = {}
    d_cluster_var = {}
    global data_list
    a=data_list[1]