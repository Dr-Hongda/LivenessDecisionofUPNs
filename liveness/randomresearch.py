import argparse, json
import random
import os
import numpy as np
import time

import sys
sys.path.append(r'/home/qhd/code/liveness/') 
from mainwithouttest import gpu_setup, train_val_pipeline, view_model_param
sys.path.append(r'/home/qhd/code/liveness/GNN/data_process/') 
from package_data_train_val import PackageData_train_val


if __name__ == '__main__':
    
    """
        USER CONTROLS, init parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details",
                        default="config/randominit.json")

    args = parser.parse_args()
    print(args.config)
    with open(args.config) as f:
        config = json.load(f)
    print(config)

    # device
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    config['net_params']['device'] = device
    # model, dataset, out_dir
    MODEL_NAME = config['model']

    DATASET_NAME = config['dataset']
    data_loc = config['data_dir']
    dataset = PackageData_train_val(data_loc, "read")

    dataset.name = DATASET_NAME
    out_dir = config['out_dir']
    # parameters
    params = config['params']

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']


    ###随机搜索参数
    param_grid = {
        'lr': [0.001, 0.005, 0.0001, 0.0005],
        'weight_decay': [0.01, 0.001, 0.0001, 0.00001],
        'h_dim': [16, 32, 64],
        'dropout': [0.4, 0.5, 0.6, 0.7, 0.8],
        'in_feat_dropout': [0.4, 0.5, 0.6, 0.7, 0.8]
    }

    MAX_EVALS = 100
    rs_validate_acc = []
    rs_validate_std = []
    # 5个参数随机组合100次，每次组合run 10次模型
    for i in range(MAX_EVALS):
        print("Random Search: {}".format(i))
        random.seed(i)
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        params['init_lr'] = hyperparameters['lr']
        params['weight_decay'] = hyperparameters['weight_decay']

        net_params['h_dim'] = hyperparameters['h_dim']
        net_params['drop_out'] = hyperparameters['dropout']
        net_params['in_feat_dropout'] = hyperparameters['in_feat_dropout']


        with open(os.path.join(out_dir, 'result1.txt'), 'a') as f:
            f.write("Random Search: " + str(i) + "; ")
            f.write("lr: " + str(params['init_lr']) + "; ")
            f.write("weight_decay: " + str(params['weight_decay']) + "; ")
            f.write("h_id: " + str(net_params['h_dim']) + "; ")
            f.write("drop_out: " + str(net_params['drop_out']) + "; ")
            f.write("in_feat_dropout: " + str(net_params['in_feat_dropout']) + "; ")
            f.write('\r\n')
        

        root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

        if not os.path.exists(out_dir + 'results'):
            os.makedirs(out_dir + 'results')
            
        if not os.path.exists(out_dir + 'configs'):
            os.makedirs(out_dir + 'configs')

        validate_result = []

        for run in range(1):
            net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
            validate_acc = train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

            validate_result.append(validate_acc)

            with open(os.path.join(out_dir, 'result1.txt'), 'a') as f:
                validate_acc *= 100
                f.write("Run "+ str(run) + ": ")
                f.write(str(dataset.name) +": ")
                f.write("validate_acc: " + str(validate_acc))
                f.write('\r\n')
        
        # 求模型迭代run次后的avg_validate_acc和std_validate_acc
        avg_validate_acc = np.mean(validate_result)
        std_validate_acc = np.std(validate_result)

        rs_validate_acc.append(avg_validate_acc)
        rs_validate_std.append(std_validate_acc)

        with open(os.path.join(out_dir, 'result1.txt'), 'a') as f:
            avg_validate_acc *= 100
            std_validate_acc *= 100
            f.write(str(dataset.name) + ": ")
            f.write("avg_validate_acc: " + str(avg_validate_acc) + "; ")
            f.write("std_validate_acc: " + str(std_validate_acc) + ";")
            f.write('\r\n')
            f.write('\r\n')


    rs_max_validate_acc = np.max(rs_validate_acc)
    rs_max_validate_index = rs_validate_acc.index(rs_max_validate_acc)
    rs_max_validate_std = rs_validate_std[rs_max_validate_index]

    with open(os.path.join(out_dir, 'result1.txt'), 'a') as f:
        rs_max_validate_acc *= 100
        f.write(str(dataset.name) + ": ")
        f.write("Random Search: " + str(rs_max_validate_index) + "; ")
        f.write("rs_max_validate_acc: " + str(rs_max_validate_acc) + "; ")
        f.write("rs_max_validate_std: " + str(rs_max_validate_std) + ";")
        f.write('\r\n')
        f.write('\r\n')
