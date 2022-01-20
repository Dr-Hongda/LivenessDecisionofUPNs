"""
    Utility file to select GraphNN model as
    selected by the user
"""
import sys
sys.path.append(r'/home/qhd/code/liveness/GNN/nets/') 

from LSTMDGNN import LSTMDGNN
# from MPNNNet import MPNNNet
# from GCNNet import GCNNet
# from MLPNet import MLPNet
# from CNNNet import CNNNet

# def MPNN(net_params):
#     return MPNNNet(net_params)

# def GN(net_params):
#     return MetaNet(net_params)

# def GCN(net_params):
#     return GCNNet(net_params)

# def MLP(net_params):
#     return MLPNet(net_params)

# def CNN(net_params):
#     return CNNNet(net_params)

def LSTMDGNN(net_params):
    return LSTMDGNN(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        # 'MPNN': MPNN,
        # 'GN' :  GN,
        # "GCN" : GCN,
        # "MLP" : MLP,
        "LSTMDGNN" : LSTMDGNN
        # "CNN" : CNN
    }
        
    return models[MODEL_NAME](net_params)