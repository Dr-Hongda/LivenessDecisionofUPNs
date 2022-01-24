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

class PackageData(torch.utils.data.Dataset):
    def __init__(self, data_loc, readorwrite, filename = None):
        #readorwrite = 1 read 0 write
        start = time.time()
        print("[I] Loading dataset %s..." % (data_loc))
        # 记得改一下
        if readorwrite == "read":
            try:
                with open(os.path.join(data_loc, filename), "rb+") as f:
                    # f = pickle.load(f)
                    f1 = pickle.load(f)
                    # f.close() 
                    self.train = f1[0]
                    self.validate = f1[1]
                    self.test = f1[2]
            except:
                print("The file is not exist")
        else:
            a = BasicDatasets(data_loc)
            x_data = a
            y_data = a.graph_labels

            # # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7)
            x_train, x_validate_test, y_train, y_validate_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 7)
            # 再将1.验证集，2.测试集，按照1：1进行随机划分
            x_validate, x_test, y_validate, y_test = train_test_split(x_validate_test, y_validate_test, test_size = 0.5, random_state = 7)
            self.train = x_train
            self.validate = x_validate
            self.test = x_test

        print('train, validate, test_net:', len(self.train), len(self.validate), len(self.test))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        # graphs, labels, name = map(list, zip(*samples))
        graphs, labels = map(list, zip(*samples))
    
        labels = torch.tensor(np.array(labels)).unsqueeze(1)

        batched_graph = []
        for i in range(len(graphs)):
            batched_graph.append(dgl.batch(graphs[i]))

        # print("=========")
        # print(batched_graph)
        # print(labels)
        # return batched_graph, labels, name
        return batched_graph, labels


def write_file(path, name):
    path = data_root
    train_file_name = name + '_train.pkl'
    test_file_name = name + '_test.pkl'

    dataset = PackageData(data_root, "write")

    start = time.time()
    with open(os.path.join(path, train_file_name) ,'wb') as f:
        pickle.dump([dataset.train,dataset.validate, dataset.test],f)
    # f.close()
    end = time.time()
    print('Time (sec):',end - start)


if __name__ == '__main__':
    save_root = '/home/qhd/code/liveness/data_generation/UPN/DS4/UPN_RG/'
    data_root = '/home/qhd/code/liveness/data_generation/UPN/DS4/UPN_RG/'

    write_file(save_root, 'DataSet4')
    
    # dir = os.getcwd() + '/data_generation/UPN/package_data/7-1/'
    # dataset = PackageData(dir)
    # print(dataset.train[0])
    
