
"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import time
import random
import glob
import argparse, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(r'/home/qhd/code/liveness/') 
from pytorchtools import EarlyStopping

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        # print("cuda current_device with GPU",torch.cuda.current_device())
        device = torch.device("cuda:" + str(gpu_id))
        print("current device is : ", device)
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
# from nets.molecules_graph_regression.load_net import gnn_model # import all GNNS

import sys
sys.path.append(r'/home/qhd/code/liveness/GNN/nets/') 
from load_net import gnn_model
from LSTMDGNN import LSTMDGNN

# from data.data import LoadData # import dataset
# from train.train_molecules_graph_regression import train_epoch, evaluate_network # import train functions

sys.path.append(r'/home/qhd/code/liveness/GNN/train/') 
from train_graph_regression import train_epoch,evaluate_network

sys.path.append(r'/home/qhd/code/liveness/GNN/data_process/') 
from package_data_train_val import PackageData_train_val



"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    # print("=======")
    # print(MODEL_NAME)
    # print(net_params)
    
    # model = gnn_model(MODEL_NAME, net_params)

    model = LSTMDGNN(net_params)
    
    print("=========")
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    

    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    # if MODEL_NAME in ['GCN', 'GAT']:
    #     if net_params['self_loop']:
    #         print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
    #         dataset._add_self_loops()
    
    trainset, validateset = dataset.train, dataset.validate

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = root_log_dir
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Nets: ", len(trainset))
    print("Validate Nets: ", len(validateset))

    # model = gnn_model(MODEL_NAME, net_params)
    model = LSTMDGNN(net_params)
    model = model.to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_F1s, epoch_val_F1s = [], [] 
    epoch_train_accs, epoch_val_accs = [], []
    
    # batching exception for Diffpool

    #不用collate batch size 设置的1
    
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    validate_loader = DataLoader(validateset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
    # initialize the early_stopping object
    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    patience = 20
    early_stopping = EarlyStopping(patience = patience, path = ckpt_dir + '/checkpoint.pt')

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                # 训练集
                start = time.time()
                epoch_train_loss, epoch_train_F1, epoch_train_acc, optimizer = train_epoch(model, optimizer, 
                                                                                           criterion, device, train_loader, epoch)

                # epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                # epoch_val_loss, epoch_val_mae,_ = evaluate_network(model, device, val_loader, epoch)
                epoch_train_losses.append(epoch_train_loss)
                # epoch_val_losses.append(epoch_val_loss)
                epoch_train_F1s.append(epoch_train_F1)
                epoch_train_accs.append(epoch_train_acc)

                # epoch_val_MAEs.append(epoch_val_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('train/_F1', epoch_train_F1, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
                # 验证集
                epoch_val_loss, epoch_val_F1, epoch_val_acc = evaluate_network(model, device, criterion, 
                                                                               validate_loader, epoch)
                
                epoch_val_losses.append(epoch_val_loss)
                epoch_val_F1s.append(epoch_val_F1)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('val/_F1', epoch_val_F1, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)

                # _, epoch_test_mae,_ = evaluate_network(model, device, test_loader, epoch)
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss="{:.4f}".format(epoch_train_loss),
                            #   train_F1="{:.4f}".format(epoch_train_F1),
                            #   validate_F1="{:.4f}".format(epoch_val_F1),
                              train_acc="{:.4f}".format(epoch_train_acc), 
                              validate_acc="{:.4f}".format(epoch_val_acc)
                              )
                # val stop
                per_epoch_time.append(time.time()-start)

                # # Saving checkpoint
                # torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                # 删除每个epoch
                # files = glob.glob(ckpt_dir + '/*.pkl')
                # for file in files:
                #     epoch_nb = file.split('_')[-1]
                #     epoch_nb = int(epoch_nb.split('.')[0])
                #     if epoch_nb < epoch - 1:
                #         os.remove(file)

                # scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(epoch_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break     
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(ckpt_dir + '/checkpoint.pt'))

    # test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    # _, test_F1, test_acc = evaluate_network(model, device, criterion, test_loader, epoch)
    _, train_F1, train_acc = evaluate_network(model, device, criterion, train_loader, epoch)
    _, validate_F1, validate_acc = evaluate_network(model, device, criterion, validate_loader, epoch)
    print("Validate F1: {:.4f}".format(validate_F1))
    print("Train F1: {:.4f}".format(train_F1))
    print("Validate acc: {:.4f}".format(validate_acc))
    print("Train acc: {:.4f}".format(train_acc))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
        FINAL RESULTS\nVALIDATE F1: {:.4f}\nTRAIN F1: {:.4f}\nVALIDATE acc: {:.4f}\nTrain acc: {:.4f}\n\n
        Total Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        np.array(validate_F1), np.array(train_F1), np.array(validate_acc)
                        , np.array(train_acc)
                        , (time.time() - t0) / 3600,
                        np.mean(per_epoch_time)))

    return validate_acc

def main():
    """
        USER CONTROLS
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details",
                        default="config/LSTMDGNN.json")

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device

    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    config['net_params']['device'] = device
    # model, dataset, out_dir
    MODEL_NAME = config['model']

    DATASET_NAME = config['dataset']
    data_loc = config['data_dir']
    dataset = PackageData_train_val(data_loc, "read", "DataSet4_train_and_validate.pkl")

    dataset.name = DATASET_NAME
    out_dir = config['out_dir']
    # parameters
    params = config['params']

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']


    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

if __name__ == '__main__':

    main()    

