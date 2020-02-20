import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# from Extrapolation-of-Learning-Curves.api import B
from api import Benchmark
# from api import Benchmark
import utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

# Read data
def cut_data(data, cut_position):
    targets = []
    for dp in data:
        targets.append(dp["Train/val_accuracy"][50])
        for tag in dp:
            if tag.startswith("Train/"):
                dp[tag] = dp[tag][0:cut_position]
    return data, targets

def read_data():
    dataset_name = 'Fashion-MNIST'
    n_configs = bench.get_number_of_configs(dataset_name)
    
    # Query API
    data = []
    for config_id in range(n_configs):
        data_point = dict()
        data_point["config"] = bench.query(dataset_name=dataset_name, tag="config", config_id=config_id)
        for tag in bench.get_queriable_tags(dataset_name=dataset_name, config_id=config_id):
            if tag.startswith("Train/"):
                data_point[tag] = bench.query(dataset_name=dataset_name, tag=tag, config_id=config_id)    
        data.append(data_point)
        
    # Split: 50% train, 25% validation, 25% test (the data is already shuffled)
    indices = np.arange(n_configs)
    ind_train = indices[0:int(np.floor(0.5*n_configs))]
    ind_val = indices[int(np.floor(0.5*n_configs)):int(np.floor(0.75*n_configs))]
    ind_test = indices[int(np.floor(0.75*n_configs)):]

    array_data = np.array(data)
    train_data = array_data[ind_train]
    val_data = array_data[ind_val]
    test_data = array_data[ind_test]
    
    # Cut curves for validation and test
    cut_position = 11
    val_data, val_targets = cut_data(val_data, cut_position)
    test_data, test_targets = cut_data(test_data, cut_position)
    train_data, train_targets = cut_data(train_data, 51)   # Cut last value as it is repeated
    
    return train_data, val_data, test_data, train_targets, val_targets, test_targets
    

#-------- new code : Sambit, 20/02/2020 ------------------------------#
# create a MLP in pytorch : 3 hidden dense layers, 16, 16, 8


class LearningCurveMLP(nn.Module):
    """
    Create the architecture
    """

    def __init__(self, input_size, dropout=0):
        """
        Default constructor
        """

        super(LearningCurveMLP, self).__init__()

        self.L1_linear = nn.Linear(7, 16)  # dense layer, 7 inputs (same as number of input features), 16 outputs, so 16 hidden units
        self.L2_linear = nn.Linear(16, 16)
        self.L3_linear = nn.Linear(16, 8)
        self.L4_linear = nn.Linear(8, 1)

        self.drpout = dropout



    def forward(self, x):
        """
        Forward propagation function
        """

        
        out1 = F.dropout(F.sigmoid(self.L1_linear(x)), p=0.3)
        out2 = F.dropout(F.sigmoid(self.L2_linear(out1)), p=0.3)
        out3 = F.dropout(F.sigmoid(self.L3_linear(out2)), p=0.3)

        out = F.sigmoid(self.L4_linear(out3))

        return out



def train(model, optimizer, criterion, clip=5):
    model.train()
    epoch_loss = []
    for val_acc, configs, targets in train_data_loader:
      optimizer.zero_grad()
      output = model([val_acc,configs])
      loss = criterion(output, targets)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
      optimizer.step()
      epoch_loss.append(loss.item())
    return np.array(epoch_loss).mean()


def evaluate(model, criterion):
  model.eval()
  epoch_loss = []
  with torch.no_grad():
    for val_acc, configs, targets in val_data_loader:
      output = model([val_acc, configs])
      loss = criterion(output, targets)
      epoch_loss.append(loss.item())
  return np.array(epoch_loss).mean()



def test(model, criterion):
    #model.load_state_dict(torch.load('content/models/model.pt'))
    model.eval()
    epoch_loss=[]
    with torch.no_grad():
      for val_acc, configs, targets in test_data_loader:
        output = model([val_acc, configs])
        loss = criterion(output, targets)
        epoch_loss.append(loss.item())
        
    return np.array(epoch_loss).mean()


def max_error(model, criterion):
    #model.load_state_dict(torch.load('content/models/model.pt'))
    model.eval()
    epoch_loss=[]
    with torch.no_grad():
      for val_acc, configs, targets in test_data_loader:
        output = model([val_acc, configs])
        loss = np.abs(output.detach().numpy()-targets.detach().numpy().reshape(-1,1))
        epoch_loss += loss.tolist()
        
    return np.array(epoch_loss)


def init_weights(m):
    for name, param in m.named_parameters():
      torch.nn.init.uniform_(param.data, -0.08, 0.08)





if __name__ == "__main__":
    print(torch.cuda.is_available())

    # bench_dir = dir_path+"fashion_mnist.json"
    bench_dir = '/home/sambit/PROGRAMMING/DL_PROJECT/TEAM_WORK_FREIBURG/Extrapolation-of-Learning-Curves/DATA/fashion_mnist.json'
    bench = Benchmark(bench_dir, cache=False)

    # get training and validation data and print shape
    train_data, val_data, test_data, train_targets, val_targets, test_targets = read_data()

    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

    device = utils.check_cuda()
    print(device)

    train_data_loader = utils.prep_data(train_data, train_targets, batch_size=32,normalization_factor_temporal_data=[100])
    val_data_loader = utils.prep_data(val_data, val_targets, batch_size=32,normalization_factor_temporal_data=[100])
    test_data_loader = utils.prep_data(test_data, test_targets, batch_size=32,normalization_factor_temporal_data=[100])



    input_size = 1
    outcome_dim = 1
    hidden_dim=35
    num_layers=2
    config_size = 7
    bidirectional = True
    lstm_dropout=0.5
    fc_dropout=0.0

    # model = UnivariatMultiStepLSTM(input_size, hidden_dim, outcome_dim, num_layers,
    #                     lstm_dropout=lstm_dropout,bidirectional=bidirectional,fc_dropout=fc_dropout)

    model = LearningCurveMLP(input_size, 0.3)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    model.apply(init_weights)


    epochs=200
    lr=0.01
    weight_decay = 10e-3
    T_0 = int(epochs/4)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


        
    train_stats = namedtuple("Stats",["train_loss", "val_loss"])
    stats = train_stats(train_loss=np.zeros(epochs),
                        val_loss=np.zeros(epochs))

    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion)
        val_loss = evaluate(model, criterion)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(),"content/models/model_5.pt")    
            print('Val loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_val_loss,val_loss))
            best_val_loss = val_loss

        print(f'Epoch: {epoch}\t Train Loss: {train_loss:.3f}\t Val. Loss: {val_loss:.3f}')
        stats.train_loss[epoch] = train_loss
        stats.val_loss[epoch] = val_loss

        scheduler.step()

    np.save("DATA/train_stats_5.npy",stats)
    

