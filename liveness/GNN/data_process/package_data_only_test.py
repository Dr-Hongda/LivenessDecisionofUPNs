#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append(r'/home/qhd/code/liveness/GNN/data_process') 
from data_loader import BasicDatasets

sys.path.append(r'/home/qhd/code/liveness/') 
import DataUtil

import torch
import time
import numpy as np
import dgl
import pickle
import os
from sklearn.model_selection import train_test_split

class PackageData_only_test(torch.utils.data.Dataset):
    def __init__(self, data_loc, readorwrite, filename = None):
        start = time.time()
        print("[I] Loading dataset %s..." % (data_loc))
        # 记得改一下
        if readorwrite == "read":
            try:
                with open(os.path.join(data_loc, filename), "rb") as f:
                    f = pickle.load(f)
                    self.data = f[0]
                    # self.validate = f[1]
                    # self.test = f[2]
            except:
                print("The file is not exist")
        else:
            a = BasicDatasets(data_loc)
            x_data = a
            y_data = a.graph_labels
            self.data = x_data
            
        print('data:', len(self.data))
        # print('train, validate, test_net:', len(self.train), len(self.validate), len(self.test))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, name = map(list, zip(*samples))
    
        labels = torch.tensor(np.array(labels)).unsqueeze(1)

        batched_graph = []
        for i in range(len(graphs)):
            batched_graph.append(dgl.batch(graphs[i]))

        # print("=========")
        # print(batched_graph)
        # print(labels)

        return batched_graph, labels, name

def write_file(path, name):
    path = data_root
    file_name = name + '.pkl'
    dataset = PackageData_only_test(data_root, "write")

    start = time.time()
    with open(os.path.join(path, file_name) ,'wb') as f:
        # pickle.dump([dataset.train,dataset.validate, dataset.test],f)
        pickle.dump([dataset.data],f)
    end = time.time()
    print('Time (sec):',end - start)

if __name__ == '__main__':
    save_root = '/home/qhd/code/liveness/data_generation/UPN/Test6/UPN_RG/'
    data_root = '/home/qhd/code/liveness/data_generation/UPN/Test6/UPN_RG/'

    
    write_file(save_root, 'Test6')
    
    # dir = '/home/qhd/code/liveness/data_generation/UPN/Test2/UPN_RG/'
    # dataset = PackageData_only_test(dir, "read")
    # print(dataset.data[0][2])
    
