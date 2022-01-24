#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch

import sys
sys.path.append(r'/home/qhd/code/liveness/GNN/layers/') 
from mlp_readout_layer import MLPReadout
from GCNLayers import GCNLayer
from BasicGNNNets import BasicNets

class GRUDGNN(BasicNets):
    def __init__(self, net_params):
        super(GRUDGNN, self).__init__(net_params)
        
        in_feat_dim = net_params['node_in_dim']
        self.reduction = torch.nn.Sequential(torch.nn.Linear(in_feat_dim, self.r_dim),
                                             torch.nn.ReLU()
                                            )         
        self.layers = nn.ModuleList([GCNLayer(self.r_dim, self.h_dim, F.relu,
                                                self.dropout, self.graph_norm, self.batch_norm, self.residual)])
        self.layers.extend([GCNLayer(self.h_dim, self.h_dim, F.relu,
                                                self.dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(self.n_layers - 1)])
        
        self.lstm = nn.GRU(self.h_dim, self.h_dim, bidirectional = False, batch_first=True)
    
        # self.layers.append(MPNN_Layer(self.in_feat_dim, out_dim, F.relu,
        #                             dropout, self.graph_norm, self.batch_norm, self.residual))
        # self.predict = torch.nn.Sequential(torch.nn.Linear(self.h_dim, self.h_dim // 2 ),
        #                                   torch.nn.ReLU(),
        #                                   torch.nn.Linear(self.h_dim // 2 , 1)
        #                                  )
        self.predict = MLPReadout(self.h_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, snorm_n = None, snorm_e = None):
        
        # print(g)
        # h = self.embedding_h(h)
        h = self.in_feat_dropout(g.ndata['feat'])

        h = self.reduction(h)

        # h = torch.zeros([g.number_of_edges(),self.h_dim]).float().to(self.device)
        # src, dst = g.all_edges()

        for conv in self.layers:
            h = conv(g, h, snorm_n)

        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        # print(hg)
        # list_hg = []
        # for i in range(len(hg)):
        #     list_hg.append(hg[i])

        # print(list_hg)
        # # hg_1 = hg.cpu()
        # # hg_1 = hg_1.detach().numpy().tolist()

        # # print(hg_1)
        # # print(len(hg_1))
        # # print(len(hg_1[0]))
        # hg_in = torch.stack(list_hg, dim=1)
        # print(hg_in)

        hg_in = hg.unsqueeze_(0)
        # print(hg_in.size())
        h_l, _ = self.lstm(hg_in)
        # print(h_l.size())
        h_p = self.predict(h_l)
        # print(h_p)
        # print(h_p.size())

        
        h_s = torch.squeeze(h_p, 0)
        # print(h_s)
        # print(h_s.size())

        out = h_s[-1].float()
        # print(out)
        # print(out.size())

        return out.sigmoid()

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        # loss = nn.L1Loss()(scores, targets)
        loss = torch.nn.CrossEntropyLoss()(scores, targets)
        return loss