import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tqdm import tqdm
import graphviz

import sys
sys.path.append(r'/home/qhd/code/liveness/') 
from ExcelToolsbyxlswriter import ExcelToolbyXlsxWriter
from DataUtil import save_data_to_json

sys.path.append(r'/home/qhd/code/liveness/obtain_graph/') 
from ArrivableGraph import get_arr_gra

def StrongDirectedGraph(node_num, flag, path):
    G = nx.DiGraph()

    list = [i for i in range(1, node_num + 1)]
    # print(list)
    nx.add_cycle(G, list)

    random_directed = int(np.random.randint(0, node_num / 2, size =1))

    for i in range(random_directed):
        start = int(np.random.randint(1, node_num + 1, size =1))
        end = int(np.random.randint(1, node_num + 1, size =1))
        if start != end:
            G.add_edge(start, end)
        else:
            continue

    labels={}
    for node in G.nodes():
        labels[node]=node

    return G

def obtain_matrix_and_M0(G):
    #初始化矩阵A1 = A+， A2 = A-
    len_p = int(len(G.nodes))
    len_t = int(len(G.edges))

    A1 = np.zeros((len_t, len_p))
    A2 = np.zeros((len_t, len_p))

    i = 0
    for edge in G.edges:
        row = i
        col1 = int(edge[0] - 1)
        col2 = int(edge[1] - 1)
        
        A2[row][col1] = 1
        A1[row][col2] = 1
        
        i += 1
    
    M0 = np.zeros((1,len_p))
    flag = int(np.random.randint(0, len_p, size = 1))
    M0[0][flag] = 1

    final_matrix = np.hstack((np.transpose(A2), np.transpose(A1),np.transpose(M0)))
    
    # print(A1)
    # print(A2)
    # print(M0)
    # print(final_matrix)
    return final_matrix

def add_transition_UPN(matrix, index):
    len_p = int(matrix.shape[0])
    insert = int((len(matrix[0]) - 1) / 2)
    colums1 = np.zeros(len_p)
    colums2 = np.zeros(len_p)
    colums2[index] = 1

    a = np.insert(matrix, insert, values = colums1, axis = 1)
    b = np.insert(a, -1, values = colums2, axis = 1)

    return b

def transform_UPN(matrix, new_transition_index):
    UPN_matrix = add_transition_UPN(matrix, new_transition_index)
    return UPN_matrix


def obtain_RG(matrix, i):
    v_list,edge_list,arctrans_list,tran_num,bound_flag = get_arr_gra(matrix, marks_upper_limit = int(matrix.shape[0]) * 4 + i * 50)
    
    modify_v_list = list(map(list, v_list))
    # print(modify_v_list)
    
    dict_rg = {}
    dict_rg['v_list'] = v_list
    dict_rg['edge_list'] = edge_list
    dict_rg['arctrans_list'] = arctrans_list
    return dict_rg

def write_UPN_matrix(matrix, i, path):
    name1 = "TN" +  str(i + 1) + "_matrix.xls"
    ET =  ExcelToolbyXlsxWriter(path, name1,'matrix')
    ET.write_xls(matrix.tolist())


def plot_arri_gra(v_list, edage_list, arctrans_list, loc):
    dot = graphviz.Digraph(format='png')
    for i in range(len(v_list)):
        dot.node("M" + str(i + 1), "M" + str(i) + "\n" + str(v_list[i]),shape="box")

    for edage,arctrans in zip(edage_list, arctrans_list):
        dot.edge(str("M" + str(edage[0] + 1)),str("M" + str(edage[1] + 1)),label= ("t" + str(arctrans+1)))
    dot.attr(fontsize='20')
    dot.format = 'png'
    try:
        dot.render(loc)
    except Exception:
        return

def plot_petri(petri_gra, loc):
    dot = graphviz.Digraph(format='png')
    data = petri_gra
    data = np.array(data,dtype=int)
    # print(data)

    deli_inde = int((data.shape[1]-1)/2)

    for i in range(len(data)):
        # dot.node("P"+str(i+1),"P"+str(i+1))
        # dot.node("P" + str(i + 1), "●",labelfloat=True)
        if data[i][-1] >= 1 :
            no_str = "P"+str(i + 1)+"\n"
            for j in range(data[i][-1]):
                no_str += "● "

            dot.node("P" + str(i + 1), no_str)
        else:
            dot.node("P" + str(i + 1), "P" + str(i + 1) + "\n\n")

    for i in range(deli_inde):
        # dot.node("P"+str(i+1),"P"+str(i+1))
        # dot.node("P" + str(i + 1), "●",labelfloat=True)
        dot.node("t" + str(i + 1), "t"+str(i + 1)+"",shape="box")
    for i in range(len(data)):
        for j in range(deli_inde):
            if data[i][j] == 1:
                dot.edge(str("P" + str(i+1)),str("t" + str(j+1)))
    # print(deli_inde)

    metrix_right = data[:,deli_inde:-1]
    # print(metrix_right)

    for i in range(len(metrix_right)):
        for j in range(metrix_right.shape[1]):
            if metrix_right[i][j] == 1:
                dot.edge(str("t" + str(j + 1)),str("P" + str(i + 1)))
    
    dot.render(loc)
    
if __name__ == '__main__':
    
    graph_path =  '/home/qhd/code/liveness/data_generation/UPN/Tnet/UPN_graph/'
    matrix_path = '/home/qhd/code/liveness/data_generation/UPN/Tnet/matrix/'
    rg_path =  '/home/qhd/code/liveness/data_generation/UPN/Tnet/UPN_RG/'

    iteration = 5000
    # times_per_type = 5
    try:
        with tqdm(range(iteration)) as t:
            for i in t:
                t.set_description('Epoch %d' % i)
                
                sdg_num = int(np.random.randint(5, 201, size =1))
                G = StrongDirectedGraph(sdg_num, i, graph_path)
                
                matrix = obtain_matrix_and_M0(G)
                add_transition_index = int(np.random.randint(1, sdg_num, size =1))
                upn_matrix = transform_UPN(matrix, add_transition_index)
                write_UPN_matrix(upn_matrix, i, matrix_path)

                new_path =  graph_path + "TN" + str(i + 1) + ".gv"
                # print(new_path)
                plot_petri(upn_matrix, loc = new_path)

                dict_Petri = {}
                dict_Petri['matrix'] = upn_matrix.tolist()
                dict_Petri['liveness'] = 1
                    
                times_per_PN = 10
                for k in range(times_per_PN):
                    name_rg = "RG" + str(k + 1)
                    dict_Petri[name_rg] = obtain_RG(upn_matrix, k)
                    
                # print(dict_Petri)
                write_path = rg_path + "TN" + str(i + 1) + '.json'
                # save_data_to_json(write_path, dict_Petri)
                save_data_to_json(write_path, dict_Petri)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from transform early because of KeyboardInterrupt')


    
    
   
    
