#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : train_graph_regression.py
# @Date    : 2020-08-27
# @Author  : mingjian
    描述
"""
"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import numpy as np
import sys
sys.path.append(r'/home/qhd/code/liveness/GNN/train/') 
# from metrics import MAE,ErrorRate
from metrics import binary_f1_score, binary_acc, ErrorRate

def train_epoch(model, optimizer, criterion, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_F1 = 0
    epoch_train_acc = 0
    epoch_train_errra = 0
    nb_data = 0
    gpu_mem = 0

    acc_score = []
    acc_target = []
    for iter, (batch_graphs, batch_targets, name) in enumerate(data_loader): #, batch_snorm_n, batch_snorm_e
        # batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        # batch_e = batch_graphs.edata['feat'].to(device)
        # batch_snorm_e = batch_snorm_e.to(device)

        batch_targets = batch_targets.float().squeeze(0).to(device)
        # batch_snorm_n = batch_snorm_n.to(device)  # num x 1
        
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs[0].to(device))
        acc_score.append(batch_scores.item())
        acc_target.append(batch_targets.item())
        # print("+++++++++")
        # print(batch_scores)exi
        # print(batch_scores.type())
        # print(batch_targets)
        # print(batch_targets.type())
        # print("+++++++++++")
        loss = criterion(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        
        # batch_predict = torch.argmax(batch_scores)
    
        nb_data += batch_targets.size(0)
        # del batch_x,batch_e
    
    # 一次求了一个网的loss，所以要把所有的求出来之后在做计算
    acc_score = np.array(acc_score)
    acc_target = np.array(acc_target)
    epoch_train_F1 += binary_f1_score(acc_score, acc_target)
    epoch_train_acc += binary_acc(acc_score, acc_target)
    epoch_loss /= (iter + 1)


    return epoch_loss, epoch_train_F1, epoch_train_acc, optimizer


def evaluate_network(model, device, criterion, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_F1 = 0
    epoch_train_acc = 0

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
            # del batch_x, batch_e
        acc_score = np.array(acc_score)
        acc_target = np.array(acc_target)
        loss = criterion(torch.from_numpy(acc_score), torch.from_numpy(acc_target))
        epoch_test_loss = loss.mean()
        epoch_test_F1 = binary_f1_score(acc_score, acc_target)
        epoch_test_acc = binary_acc(acc_score, acc_target)

    return epoch_test_loss, epoch_test_F1, epoch_test_acc