# This is a sample Python script.

# Token:
# cd5c0f2b8bb97f0cd4e46dd6f0e5647922a163d3

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool
# windows, actions, and settings.
import sys
import os
import math
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stem_plot as st
import datetime as dt
import seaborn as sns
# numm=-math.inf
# if numm<10000000:print (math.atan(numm))
# exit()
import xlsxwriter

DEBUG1 = True
DEBUG2 = False
DEBUG3 = False
DEBUG4 = False
DEBUG5 = False


def get_logger(name=__file__, file='log.txt', encoding='utf-8'):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] %(filename)s:%(lineno)d %(levelname)-8s %(message)s')
    os.remove("log.txt")
    fh = logging.FileHandler(file, encoding=encoding, mode='w')
    #    fh = logging.FileHandler(file, encoding=encoding)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)

    return log


log = get_logger()

log.info("start")


# print = lambda text="": log.debug(text)
# print("lambda")
# exit()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# ['Num','Date','MinTemp','RayFrom','RayTo','dx','dy','K','B','FLiine']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # TODO # почиctить фильтрацию
    input_file_name = 'daily-min-temperatures.csv'
    full_df = pd.read_csv(
        input_file_name,
        names=[
            'Date',
            'MinTemp'],
        index_col=[0])
    start_date = "1981-01-01"
    end_date = "1981-01-12"
    df = full_df.loc[start_date:end_date].copy(deep=True)

    # df['1983-01-01':'1983-02-01']

    print(df, df.dtypes, df.index)
    # df['1983-01-01':'1983-02-01']
    data_list = df['MinTemp'].tolist()
    # exit()

    # between_two_dates = after_start_date & before_end_date
    # filtered_dates = df.loc[between_two_dates]
    # # list_of_MinTemp = df['MinTemp']
    # list_of_dates = df['1983-01-01':'1983-01-11']
    # print(list_of_dates.index)
    # #exit()
    # wdf=pd.DataFrame({'Date':pd.to_datetime(list_of_dates),'MinTemp':list_of_MinTemp})
    # mask = (df['Date'] > '1083-6-1') & (df['Date']<= '1984-6-10')
    # df.query(mask)
    # #wdf=df.set_index('Date')
    # #filtered_df = df[df["Date"].isin(pd.date_range('1983-01-01', '1983-02-05'))]
    # filtered_df = df.loc[df["Date"].between('1983-06-1', '1983-02-05')]
    # print(filtered_df)
    # print(wdf.dtypes)
    # filtered_df = wdf.loc['19830101':'19830131']
    # print(wdf)
    # exit()
    # df1 = df.copy(deep=True)
    # start_date = "1983-1-1"
    # end_date = "1984-1-31"
    # after_start_date = df["Date"] >= start_date
    # before_end_date = df["Date"] <= end_date
    # df1 = df['19830101':'19830131']
    # print(df1)
    # df.query('19830101 < df[Date < 19840201')
    # df1 = df[(df['Date'] > '19840101') & (df['Date'] < '19850201')]
    # df1 = df[start_date:end_date]
    # df.set_index(df['Date'])
    # df1 = df['19830101':'19830131']
    # print(df)
    # exit()
    # list_of_MinTemp = df['MinTemp']
    # list_of_dates = df['Date']
    # print(list_of_dates)
    # print(df)
    # print(list_of_dates.dtypes)
    # print(list_of_dates)
    # print(list_of_MinTemp)
    # print(list_of_dates.dtypes)
    # list_of_dates = pd.to_datetime(list_of_dates)
    # print(list_of_dates.dtypes)
    # ##work_df=pd.DataFrame("Name":list_of_MinTemp,'Joined_data':)
    # work_df = pd.DataFrame({'Date': pd.to_datetime(list_of_dates), "MinTemp": list_of_MinTemp})
    #
    # # filtered_df = df.query("MinDate  >= '1990-06-1' and MinDate <='1990-02-05'")
    #
    # print(work_df, work_df.dtypes)
    # exit()
    # df.date=df.Date.df.date.dt.date
    # selector=df['Date'].dt.date.astype(str) == "1981.01.01"
    # print(df['Date'].dt.date.astype(str),selector)
    # print(df[df['Date'].dt.date.astype(str) == '2017-03-20'])]

    # start_date = "1983-01-01"
    # end_date = "1984-01-31"
    #
    # after_start_date = df["Date"] >= start_date
    # before_end_date = df["Date"] <= end_date
    # between_two_dates = after_start_date & before_end_date
    # filtered_dates = df.loc[between_two_dates]
    #
    # exit()
# TODO сделать одномерный массив data_list заполнить для замены дат
# data_list = np.array([[],[]],dtype=np.intp)
data_list = df['MinTemp'].tolist()
# ddd = np.array(data_list).tolist()
# print("***DDD***:",type(ddd),type(data_list))

# data_temp = df['Index']
# pd.DataFrame(data_list).to_csv('data_list-{}.csv'.format(max_dim))
max_dim = len(df)
out_data_list_name = "data_list from {0} to {1} length {2}.csv".format(
    start_date, end_date, max_dim)
new_ind = np.arange(0, max_dim)

st.stemplot(new_ind, data_list)
pd.DataFrame(data_list).to_csv(out_data_list_name)
full_df.to_csv("full_df.csv")


# TODO построить
# tips_df = sns.load_dataset('tips')
# tips_df.head()
# plt.show()


#
# TODO замена дат порядковыми номероми
# for ind in range(0,len(df)):
#     new_ind.append(ind)
#     print("new_ind:", ind, new_ind[ind])
# data_list[:,0]=new_ind
# if DEBUG:
#     for ind in range(0,len(df)): print("DEBUG_3:", ind, df)

def line(x0, x1, data_list):
    y0 = data_list[x0]
    y1 = data_list[x1]

    k = (y1 - y0) / (x1 - x0)
    B = (x1 * y0 - x0 * y1) / (x1 - x0)
    # eps = 1.e-3
    # return [k, B, k * x + B, (k * x + B - data_list[x]) <= 0]
    return k


# def cluster_fill(x0):


def is_visible(x0, x1, x):
    y0 = data_list[x0]
    y1 = data_list[x1]

    k = (y1 - y0) / (x1 - x0)
    B = (x1 * y0 - x0 * y1) / (x1 - x0)
    # eps = 1.e-3
    return (k * x + B - data_list[x] <= 0)


# max_dim = sum(1 for my_line in open(input_file_name,'r'))

def print_graph_array(graph_array):
    for ind in range(0, len(graph_array)):
        print(ind, ":", graph_array[ind])


def is_cluster_found(x0, x1, x2, data_list, max_dim, control_k):
    vis_k = line(x0, x1, data_list)
    try:
        vis_k_next = line(x0, x2, data_list)
    except IndexError as e:
        # print(e)
        # print(sys.exc_info())
        vis_k_next = vis_k
    ind = np.argmax([control_k, vis_k, vis_k_next])
    return vis_k, vis_k_next, ind


logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')


def build_graph_with_clusters(start_point, data_list, max_dim, DEBUG=False):
    x0 = start_point  # начало кластера
    cluster_start = start_point  # начало кластера
    x1 = x0 + 1  # рабочая вершина
    x2 = x1 + 1  # контрольная вершина
    cluster_start = x0
    cluster_size = 0
    clusters_count = 0
    # cluster_start.append(start_point)
    graph_array = np.eye(max_dim)
    graph_array.fill(0)
    control_k = line(x0, x1, data_list)
    log.info("x0={} x1={}, x2={} control01={} ".format(x0, x1, x2, control_k))
    if max_dim == 2:  # если всего 2 вершины, матрица 2x2

        vis_k, vis_k_next, ind = is_cluster_found(
            x0, x1, x2, data_list, max_dim)
        log.debug(
            "1=Start cluster x0={} (x1={} x2={}) ind={} vis_k=({} {})".format(
                x0, x1, x2, ind, vis_k, vis_k_next))

        if ind == 2:  # end cluster x0
            graph_array[x0][x0] = 2
            # array_cluster_size[x0]
        else:
            graph_array[x1][x1] = 2
            # graph_array[x0][x0] = 1
            # graph_array[x1][x0] = 1
            return graph_array
        if max_dim == 3:  # если матрица 3x3
            vis_k, vis_k_next, ind = is_cluster_found(
                x0, x1, x2, data_list, max_dim)
            if ind == 0:  # end cluster x0
                log.debug(
                    "2=Start cluster x0={} (x1={} x2={}) ind={} vis_k=({} {})".format(
                        x0, x1, x2, ind, vis_k, vis_k_next))

                graph_array[x0][x0] = 2

            # graph_array.fill(1)
            # graph_array[x1][x1]=2
            return graph_array
    array_k = []
    array_cluster_start = np.arange(0, max_dim)
    array_cluster_sizes = np.arange(0, max_dim)
    interior_list = []
    array_cluster_sizes.fill(0)
    cluster_list = array_cluster_sizes.tolist()

    clusters_count = 0
    cluster_len = 0
    control_x = x1
    # -----------------------------------------------
    # Start main algoriithm
    # -----------------------------------------------
    control_k = line(x0, x1, data_list)
    control_2 = line(x0, x2, data_list)
    x0=6
    x1 = 7
    while True:
        # x1 = x0 + 1
        x2 = x1 + 1

        cluster_len = 0
        while x2 <= max_dim - 1:

            # x2 = x1+1
            vis_k, vis_k_next, ind = is_cluster_found(
                x0, x1, x2, data_list, max_dim, control_k)
            control_2 = line(x0, x2, data_list)

            log.info(
                "x0={} x1={}, x2={} ind={} control_k={} control_2={}".format(
                    x0, x1, x2, ind, control_k, control_2))
            # cluster end x0 start new cluster x1=x2
            # control_k=libe(x0,x1,data_list
            if ind == 2:
                st.rays_plot(x0, data_list[x0], x2, data_list[x2])

                # if (x2 - x1 == 1) and cluster_len == 0:
                # TODO Fix patch #1
                if (x2 - x1 == 1):
                # if len(interior_list) == 0 and x1 == x0 + 1:
                    # ---------------------------
                    interior_list.append(x1)
                    list(array_cluster_sizes).append(interior_list)
                    cluster_list[x0] = interior_list
                    # test_list[0] = [2,3]
                    array_cluster_sizes[x0] = x2-x1+1
                # ---------------------------

                log.debug(
                    "ind=2:****** Start={}{} sublist={}".format(x0, x2, interior_list))
                # array_cluster_sizes[x0].append(cluster_list)
                # x0 = x1  # новое начало кластера x0=x1
                clusters_count += 1
                array_cluster_sizes[x1] += 1
                # TODO Fix patch #2

                cluster_list[x1] = interior_list

                x1 = x2
                x2 += 1
                control_k = line(x0, x1, data_list)
                control_2 = line(x0, x2, data_list)
                log.info(
                    "x0={} x1={}, x2={} ind={} control_k={} control_2={}".format(
                        x0, x1, x2, ind, control_k, control_2))
                log.debug("5 Cluster start={} interior_list={} cluster_len={}".format(x1, interior_list, cluster_len))


                interior_list = []
                # x1 = x0 + 1
                x2 = x1 + 1
                print("**0****", x0, x1, x2, control_k, ind)
                # array_k.append([x0, x1, x2, vis_k, vis_k_next, cluster_len])

                if DEBUG1:
                    print(
                        "DEBUG_1:From= ",
                        x0,
                        "To= ",
                        x1,
                        "k= ",
                        vis_k,
                        "Next= ",
                        x2,
                        "next_k=",
                        vis_k_next,
                        "cluster_size=",
                        cluster_size,
                        "ind=",
                        ind,
                        "cluster_len=",
                        cluster_len)
            if ind != 2:  # continue cluster x0
                log.debug(
                    "4=Start cluster x0={} (x1={} x2={}) ind={} vis_k=({} {} increase cluster length x0={})".format(
                        x0,
                        x1,
                        x2,
                        ind,
                        vis_k,
                        vis_k_next,
                        x0))
                interior_list.append(x2)
                cluster_len += 1
                log.debug("5 Cluster start={} interior_list={}".format(x1, interior_list))
                if len(array_cluster_sizes) == 0:
                    array_cluster_sizes.append(cluster_len)
                array_cluster_sizes[x1] += 1
                # x1=x2
                # control_k=line(x0,x1,data_list)
                x2 += 1
                print("**1****", x0, x1, x2, ind, cluster_len, interior_list)
                # if DEBUG1: print("DEBUG_1:From= ", x0, "To= ", x1, "k= ", vis_k, "Next= ", x2, "next_k=", vis_k_next,
                #                  "cluster_size=", cluster_size, "ind=", ind, "cluster_len=", cluster_len)
            array_k.append([x0, x1, x2, vis_k, vis_k_next, cluster_len])
            # x1 += 1
            # x2 += 1
        if x2 > (max_dim - 1):
            cluster_len = x2 - x1
            array_cluster_sizes[x1] = cluster_len
            cluster_list[x1] = interior_list
            print("**111****", x0, x1, x2, ind, cluster_len, interior_list)
            print("------------------------------------------------------------")
            break

            is_growing = vis_k <= vis_k_next
            is_decrease = not is_growing
            if is_growing:
                # Вершина x2 вхоит в кластер? -да
                graph_array[x0][x1] = 1
                # graph_array[x0][x2] = 1
                graph_array[x1][x0] = 1
                # graph_array[x2][x0] = 1
                if DEBUG4:
                    print(
                        "DEBUG_4:From= ",
                        x0,
                        "To= ",
                        x1,
                        "k= ",
                        vis_k,
                        "Next= ",
                        x2,
                        "next_k=",
                        vis_k_next,
                        "cluster_size=",
                        cluster_size,
                        is_growing)

                # x0 = x1
                x1 = x1 + 1
                x2 = x1 + 1
                # TODO  graph_array[x0][x0] += 1
                graph_array[x0][x0] += 1
                # val=graph_array[x0][x0]
                # cluster_range=range(x0,val)
                # for i in range(x0,x0+cluster_range-1):
                #     for j in range(x0, x0+cluster_range-1):
                #             graph_array[i][j]=1
                # cluster_size = cluster_size + 1
                # graph_array[x1][x0] = 1
            else:
                # Вершина x1 вхоит в x0 кластер? -нет
                graph_array[x0][x0] = cluster_size
                cluster_size = 1
                # начало нового кластера в x1
                x0 = x1
                x1 = x0 + 1
                x2 = x1 + 1
                # TODO Проверить логику
                # graph_array[x0][x1] = 1
                # graph_array[x1][x0] = 1
                x1 = x0 + 1
                x2 = x1 + 1
                if DEBUG2:
                    print(
                        "DEBUG_2:",
                        x0,
                        x1,
                        x2,
                        vis_k,
                        vis_k_next,
                        cluster_size,
                        is_decrease)
        # if x0 == max_dim - 3:
        #     #graph_array[x0][x0] = cluster_size+1
        #     #graph_array[x0][x1] = 1
        #     #graph_array[x0][x2] = 1
        #     #graph_array[x1][x0] = 1
        #     #graph_array[x2][x0] = 1
        #     #graph_array[x2][x0] = 1
        #     break
        # if x1 == max_dim - 1:
        #     break
    if DEBUG:
        lenk = len(array_k)
        print("****  Array_k  ****")
        for i in range(0, lenk):
            print("DEBUG_5: ind=", i, array_k[i])
        print("******************")
        print("****  Array_cluster_size  ({}) ****".format(clusters_count))
        for i in range(0, len(array_cluster_sizes)):
            print("DEBUG_5: ind=", i, array_cluster_sizes[i])
        print("******************")
    print(cluster_list)
    return graph_array



graph_array = build_graph_with_clusters(0, data_list, max_dim, True)

plt.show()
# print(graph_array)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# x = ['C', 'C++', 'Java', 'Python', 'PHP']
# y = [4,2,3,7,3]
# ax.bar(x,y)
plt.matshow(graph_array)

plt.show()

# eOutput = pd.DataFrame(graph_array)
# writer = pd.ExcelWriter('ArrayFromPycharm.xlsx', engine='xlsxwriter')
#
# # Convert the dataframe to an XlsxWriter Excel object.
# eOutput.to_excel(writer, sheet_name='Sheet1', index=False)
#
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

# ---------------------------------  Version 1.0 ------------------------------
