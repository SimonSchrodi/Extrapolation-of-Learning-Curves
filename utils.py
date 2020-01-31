import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def check_cuda():
    """Returns device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def make_torch_dataset(data, targets):
    return TensorDataset(*data,targets)

def make_torch_dataloader(torch_dataset,batch_size):
    return DataLoader(dataset=torch_dataset,batch_size=batch_size)

def remove_config_entry(configs,key):
    for c in configs:
      if key in c.keys():
        del c[key]
    return configs

def get_first_n_epochs(temporal_data,n=10):
    return temporal_data[:,:n]

def extract(data, key):
    output = []
    for d in data:
        output.append(d[key])
    return np.array(output)
