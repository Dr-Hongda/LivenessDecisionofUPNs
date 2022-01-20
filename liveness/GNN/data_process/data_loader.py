#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : data_loader
# @Author  : qhd
"""
import torch
import torch.utils.data

import sys
sys.path.append(r'/home/qhd/code/liveness/') 
from DataUtil import save_data_to_json
import DataUtil as DU
import dgl
import os

from pathlib import Path
import json
import torch.nn as nn

def parse_dir(root_dir):
        path = Path(root_dir)
        all_json_file = list(path.glob('**/*.json'))
        parse_result = []
        i = 0
        for json_file in all_json_file:
            # 获取所在目录的名称
            service_name = json_file.name
            print(service_name[: -5])
            with json_file.open() as f:
                json_result = json.load(f)
                json_result['filename'] = service_name[: -5]
                json_result['index'] = i
                i += 1
                parse_result.append(json_result)
        return parse_result

class BasicDatasets(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        #读取该文件夹下所有的json数据
        self.data_dir = data_dir
        # self.alldata = parse_dir(data_dir)
        self.name = []
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = 0
        self._prepare(data_dir)

    def _prepare(self, data_dir):
        
        path = Path(data_dir)
        all_json_file = list(path.glob('**/*.json'))

        self.n_samples = len(all_json_file)
        print("preparing %d nets for the set..." % (len(all_json_file)))
        
        i = 0
        for json_file in all_json_file:
            # 获取所在目录的名称
            service_name = json_file.name
            print(service_name[: -5])
            with json_file.open() as f:
                json_result = json.load(f)
                json_result['filename'] = service_name[: -5]
                self.name.append(str(json_result['filename']))
                json_result['index'] = i
                i += 1
                
                self.graph_labels.append(float(json_result['liveness']))
                all_rg_a_net = []
                
                for i in range(10):
                    
                    graph_name = "RG" + str(i + 1)
                    # Create the DGL Graph
                    g = dgl.DGLGraph()
                
                    g.add_nodes(len(json_result[graph_name]['v_list']))
                    
                    len_v = len(json_result[graph_name]['v_list'][0])
                    # print(len_v)
                    #补齐向量长度, 注意最大长度
                    add_zero = 210 - len_v
                    pad = nn.ZeroPad2d(padding = (0, add_zero, 0, 0))

                    a = torch.tensor(json_result[graph_name]['v_list']).float()
                    g.ndata['feat'] = pad(a).float()

                    # a = torch.tensor(data[graph_name]['v_list']).float()
                    # add_zero = 60 - len_v
                
                    # g.ndata['feat'] = nn.ZeroPad2d(padding = (0, add_zero, 0, 0))
                    # g.ndata['feat'] = torch.tensor(data[graph_name]['v_list']).float()

                    for src, dst in json_result[graph_name]['edge_list']:
                        g.add_edges(src, dst)
                    g.edata['feat'] = torch.tensor(json_result[graph_name]['arctrans_list']).view(-1,1).float()
                    
                    g = dgl.add_self_loop(g)
                    all_rg_a_net.append(g)

                self.graph_lists.append(all_rg_a_net)


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx], self.name[idx]

    


if __name__ == '__main__':
    dir = '/home/qhd/code/liveness/data_generation/UPN/Test2/UPN_RG/'
    a = BasicDatasets(dir)
    
    
    print(len(a))
    
    print(a[0][1])

