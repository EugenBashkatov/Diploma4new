import numpy as np

def get_index_of_max(par_list):

    index_of_max = np.argmax(list(par_list))
    return index_of_max

def get_data_by_index(par_dataframe,par_index):
    res_data = par_dataframe.index[par_index]
    return res_data

def display_cluster_to_matrix(par_dict,graph_array):
    for k in par_dict.values():
        # print("k=",k)
        for j in k:
            # print(j)
            graph_array[k[0]][j] = 1
            graph_array[j][k[0]] = 1
    # print(graph_array)
    return True

def statistics(par_input):
    res_var = np.var(par_input)
    res_mean = np.mean(par_input)
    return res_mean, res_var


