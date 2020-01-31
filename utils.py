import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import numpy as np
import typing

def check_cuda():
    """Returns device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def tt(ndarray:np.ndarray):
  if torch.cuda.is_available():
    return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=True)
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=True)

def make_torch_dataset(data:np.ndarray, targets:np.ndarray)->torch.utils.data.dataset:
    return TensorDataset(*data,targets)

def make_torch_dataloader(torch_dataset:torch.utils.data.dataset,batch_size)->torch.utils.data.dataset:
    return DataLoader(dataset=torch_dataset,batch_size=batch_size)

def remove_config_entry(configs:np.ndarray,keys=['activation',
                                                 'cosine_annealing_T_max',
                                                 'cosine_annealing_eta_min',
                                                 'imputation_strategy',
                                                 'learning_rate_scheduler',
                                                 'loss',
                                                 'mlp_shape',
                                                 'normalization_strategy',
                                                 'optimizer',
                                                 'network'])->np.ndarray:
    for c in configs:
        for key in keys:
            if key in c.keys():
                del c[key]
    return configs

def get_first_n_epochs(temporal_data:np.ndarray,n=10):
    return temporal_data[:,:n]

def extract_from_data(data:np.ndarray, key)->np.ndarray:
    output = []
    for d in data:
        output.append(d[key])
    return np.array(output)

def normalize_configs(configs:np.ndarray)->np.ndarray:
    output = []
    for config in configs:
        #config['activation'] = 0 if "relu" else 1
        config['batch_size'] /= 511
        #config['cosine_annealing_T_max'] = 1
        #config['cosine_annealing_eta_min'] = config['cosine_annealing_eta_min']
        #config['imputation_strategy'] = 0
        config['learning_rate'] = config['learning_rate']
        #config['learning_rate_scheduler'] = 0
        #config['loss'] = 0
        config['max_dropout'] = config['max_dropout']
        config['max_units'] /= 1024
        #config['mlp_shape'] = 0
        config['momentum'] = config['momentum']
        #config['normalization_strategy'] = 0
        config['num_layers'] /= 4
        #config['optimizer'] = 0
        config['weight_decay'] = config['weight_decay']

        list_values = [float(v) for v in config.values()]
        output.append(list_values)

    return np.array(output)

def normalize_accuracies(accuracies:np.ndarray)->np.ndarray:
    return accuracies/100

def prep_data(data:np.ndarray, target_data:np.ndarray, batch_size, temporal_keys=['Train/val_accuracy'], first_n_epochs=10):
    configs = extract_from_data(data,"configs")
    configs = remove_config_entry(configs)
    configs = normalize_configs(configs)
    configs = torch.FloatTensor(configs)

    data_list = []
    for k in temporal_keys:
        d = extract_from_data(data,key=k)
        d = get_first_n_epochs(d, first_n_epochs)
        d = torch.FloatTensor(d)
        data_list.append(d)

    data_list.append(configs)
    target_data = torch.FloatTensor(target_data)
    dataset = make_torch_dataset(data_list,target_data)
    data_loader = make_torch_dataloader(dataset,batch_size)
    return data_loader

