##################################################################
#      test the data outside the dataset by the trained models   #
##################################################################
import pickle
import argparse
import json
import os
from torch.utils.data import DataLoader
import torch
import random

import sys
sys.path.append(r'/home/qhd/code/liveness/GNN/nets/') 
from LSTMDGNN import LSTMDGNN

sys.path.append(r'/home/qhd/code/liveness/GNN/data_process/') 
from package_data_only_test import PackageData_only_test

sys.path.append(r'/home/qhd/code/liveness/GNN/train/') 
from metrics import binary_f1_score, binary_acc, ErrorRate

sys.path.append(r'/home/qhd/code/liveness/') 
from main import gpu_setup

import torch.optim as optim
import time
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def evaluate_network(model, device, criterion, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_F1 = 0
    epoch_train_acc = 0
    wrong_list = []
    # epoch_test_errra = 0
    with torch.no_grad():
        acc_score = []
        acc_target = []
        for iter, (batch_graphs, batch_targets, name) in enumerate(data_loader):
            # batch_x = batch_graphs.ndata['feat'].to(device)
            # batch_e = batch_graphs.edata['feat'].to(device)
            # batch_snorm_e = batch_snorm_e.to(device)
            batch_targets = batch_targets.item()
            # batch_snorm_n = batch_snorm_n.to(device)
            batch_scores = model.forward(batch_graphs[0].to(device))
            acc_score.append(batch_scores.item())
            acc_target.append(batch_targets)

            print("+++++++++++++")
            print(name)
            print(batch_scores.item())
            print(batch_targets)
            print("+++++++++++++")

            if int(batch_targets) == 0 and np.double(batch_scores.item()) > 0.5 :
                wrong_list.append(name)
            if int(batch_targets) == 1 and np.double(batch_scores.item()) <= 0.5 :
                wrong_list.append(name)

            # del batch_x, batch_e
        acc_score = np.array(acc_score)
        acc_target = np.array(acc_target)
        loss = criterion(torch.from_numpy(acc_score), torch.from_numpy(acc_target))
        epoch_test_loss = loss.mean()
        epoch_test_F1 = binary_f1_score(acc_score, acc_target)
        epoch_test_acc = binary_acc(acc_score, acc_target)

    return epoch_test_loss, epoch_test_F1, epoch_test_acc, wrong_list


if __name__ == '__main__':
    
    #初始化
    with open("config/LSTMDGNN.json") as f:
        config = json.load(f)

    #device
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    config['net_params']['device'] = device
    
    #parameters
    params = config['params']

    #network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']


    # #初始化模型和optimizer
    model = LSTMDGNN(net_params)
    model = model.to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

    start = time.time()
    # 加载模型
    # model_path = '/home/qhd/code/liveness/data_generation/UPN/package_data/DS4/random/checkpoints/LSTMDGNN_DataSet4_GPU0_14h11m59s_on_Sep_18_2021/RUN_/'
    model_path = '/home/qhd/code/liveness/data_generation/UPN/package_data/DS4/checkpoints/LSTMDGNN_DataSet4_GPU0_19h47m54s_on_Sep_18_2021/RUN_/'
    model.load_state_dict(torch.load(model_path + '/checkpoint.pt'))
    # print(model)
    
    # 加载数据
    data_path = '/home/qhd/code/liveness/data_generation/UPN/Test6/UPN_RG/'
    test_dataset = PackageData_only_test(data_path, "read", "Test6.pkl")
    data_loader = DataLoader(test_dataset.data, batch_size=params['batch_size'], shuffle=False, collate_fn=test_dataset.collate)

    # 输出结果
    print("************************")
    # _, test_F1, test_acc = evaluate_network(model, device, criterion, test_loader, epoch)
    _, data_F1, data_acc, wrong_list = evaluate_network(model, device, criterion, data_loader)
    print("data F1: {:.4f}".format(data_F1))
    print("data acc: {:.4f}".format(data_acc))
    print("wrong data:" + str(wrong_list))
    print("************************")

    end = time.time()
    print('Time (sec):',end - start)

