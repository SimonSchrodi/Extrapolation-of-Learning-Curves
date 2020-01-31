import torch
from torch.utils.data import TensorDataset, DataLoader

def check_cuda():
    """Returns device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def make_torch_dataset(data, targets):
    return TensorDataset(data,targets)

def make_torch_dataloader(torch_dataset,batch_size):
    return DataLoader(dataset=torch_dataset,batch_size=batch_size)

def filter_config(configs):
    return configs[:]["Train/val_accuracy"][:10]

