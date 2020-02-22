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
from torch.optim import Adam, SGD
from torchsummary.torchsummary import summary
# import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
# import numpy as np
import typing
import os
import json


class Benchmark():
    """API for TabularBench."""

    def __init__(self, data_dir, cache=False, cache_dir="cached/"):
        """Initialize dataset (will take a few seconds-minutes).

        Keyword arguments:
        bench_data -- str, the raw benchmark data directory
        """
        if not os.path.isfile(data_dir) or not data_dir.endswith(".json"):
            raise ValueError("Please specify path to the bench json file.")

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache = cache

        print("==> Loading data...")
        self.data = self._read_data(data_dir)
        self.dataset_names = list(self.data.keys())
        print("==> Done.")

    def query(self, dataset_name, tag, config_id):
        """Query a run.

        Keyword arguments:
        dataset_name -- str, the name of the dataset in the benchmark
        tag -- str, the tag you want to query
        config_id -- int, an identifier for which run you want to query, if too large will query the last run
        """
        config_id = str(config_id)
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")

        if config_id not in self.data[dataset_name].keys():
            raise ValueError("Config nr %s not found for dataset %s." % (config_id, dataset_name))

        if tag in self.data[dataset_name][config_id]["log"].keys():
            return self.data[dataset_name][config_id]["log"][tag]

        if tag in self.data[dataset_name][config_id]["results"].keys():
            return self.data[dataset_name][config_id]["results"][tag]

        if tag in self.data[dataset_name][config_id]["config"].keys():
            return self.data[dataset_name][config_id]["config"][tag]

        if tag == "config":
            return self.data[dataset_name][config_id]["config"]

        raise ValueError("Tag %s not found for config %s for dataset %s" % (tag, config_id, dataset_name))

    def query_best(self, dataset_name, tag, criterion, position=0):
        """Query the n-th best run. "Best" here means achieving the largest value at any epoch/step,

        Keyword arguments:
        dataset_name -- str, the name of the dataset in the benchmark
        tag -- str, the tag you want to query
        criterion -- str, the tag you want to use for the ranking
        position -- int, an identifier for which position in the ranking you want to query
        """
        performances = []
        for config_id in self.data[dataset_name].keys():
            performances.append((config_id, max(self.query(dataset_name, criterion, config_id))))

        performances.sort(key=lambda x: x[1] * 1000, reverse=True)
        desired_position = performances[position][0]

        return self.query(dataset_name, tag, desired_position)

    def get_queriable_tags(self, dataset_name=None, config_id=None):
        """Returns a list of all queriable tags"""
        if dataset_name is None or config_id is None:
            dataset_name = list(self.data.keys())[0]
            config_id = list(self.data[dataset_name].keys())[0]
        else:
            config_id = str(config_id)
        log_tags = list(self.data[dataset_name][config_id]["log"].keys())
        result_tags = list(self.data[dataset_name][config_id]["results"].keys())
        config_tags = list(self.data[dataset_name][config_id]["config"].keys())
        additional = ["config"]
        return log_tags + result_tags + config_tags + additional

    def get_dataset_names(self):
        """Returns a list of all availabe dataset names like defined on openml"""
        return self.dataset_names

    def get_openml_task_ids(self):
        """Returns a list of openml task ids"""
        task_ids = []
        for dataset_name in self.dataset_names:
            task_ids.append(self.query(dataset_name, "OpenML_task_id", 1))
        return task_ids

    def get_number_of_configs(self, dataset_name):
        """Returns the number of configurations for a dataset"""
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")
        return len(self.data[dataset_name].keys())

    def get_config(self, dataset_name, config_id):
        """Returns the configuration of a run specified by dataset name and config id"""
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")
        return self.data[dataset_name][config_id]["config"]

    def plot_by_name(self, dataset_names, x_col, y_col, n_configs=10, show_best=False, xscale='linear', yscale='linear',
                     criterion=None):
        """Plot multiple datasets and multiple runs.

        Keyword arguments:
        dataset_names -- list
        x_col -- str, tag to plot on x-axis
        y_col -- str, tag to plot on y-axis
        n_configs -- int, number of configs to plot for each dataset
        show_best -- bool, weather to show the n_configs best (according to query_best())
        xscale -- str, set xscale, options as in matplotlib: "linear", "log", "symlog", "logit", ...
        yscale -- str, set yscale, options as in matplotlib: "linear", "log", "symlog", "logit", ...
        criterion -- str, tag used as criterion for query_best()
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if not isinstance(dataset_names, (list, np.ndarray)):
            raise ValueError("Please specify a dataset name or a list list of dataset names.")

        n_rows = len(dataset_names)
        fig, axes = plt.subplots(n_rows, 1, sharex=False, sharey=False, figsize=(10, 7 * n_rows))

        if criterion is None:
            criterion = y_col

        loop_arg = enumerate(axes.flatten()) if len(dataset_names) > 1 else [(0, axes)]

        for ind_ax, ax in loop_arg:
            for ind in range(n_configs):
                try:
                    if ind == 0:
                        instances = int(self.query(dataset_names[ind_ax], "instances", 0))
                        classes = int(self.query(dataset_names[ind_ax], "classes", 0))
                        features = int(self.query(dataset_names[ind_ax], "features", 0))

                    if show_best:
                        x = self.query_best(dataset_names[ind_ax], x_col, criterion, ind)
                        y = self.query_best(dataset_names[ind_ax], y_col, criterion, ind)
                    else:
                        x = self.query(dataset_names[ind_ax], x_col, ind + 1)
                        y = self.query(dataset_names[ind_ax], y_col, ind + 1)

                    ax.plot(x, y, 'p-')
                    ax.set_xscale(xscale)
                    ax.set_yscale(yscale)
                    ax.set(xlabel="step", ylabel=y_col)
                    title_str = ", ".join([dataset_names[ind_ax],
                                           "features: " + str(features),
                                           "classes: " + str(classes),
                                           "instances: " + str(instances)])
                    ax.title.set_text(title_str)
                except ValueError:
                    print("Run %i not found for dataset %s" % (ind, dataset_names[ind_ax]))
                except Exception as e:
                    raise e

    def _cache_data(self, data, cache_file):
        os.makedirs(self.cache_dir, exist_ok=True)
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def _read_cached_data(self, cache_file):
        with gzip.open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def _read_file_string(self, path):
        """Reads a large json string from path. Python file handler has issues with large files so it has to be chunked."""
        # Shoutout to https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file
        file_str = ''
        with open(path, 'r') as f:
            while True:
                block = f.read(64 * (1 << 20))  # Read 64 MB at a time
                if not block:  # Reached EOF
                    break
                file_str += block
        return file_str

    def _read_data(self, path):
        """Reads cached data if available. If not, reads json and caches the data as .pkl.gz"""
        cache_file = os.path.join(self.cache_dir, os.path.basename(self.data_dir).replace(".json", ".pkl.gz"))
        if os.path.exists(cache_file) and self.cache:
            print("==> Found cached data, loading...")
            data = self._read_cached_data(cache_file)
        else:
            print("==> No cached data found or cache set to False.")
            print("==> Reading json data...")
            data = json.loads(self._read_file_string(path))
            if self.cache:
                print("==> Caching data...")
                self._cache_data(data, cache_file)
        return data


# Read data
def cut_data(data, cut_position):
    targets = []
    for dp in data:
        targets.append(dp["Train/val_accuracy"][50])
        for tag in dp:
            if tag.startswith("Train/"):
                dp[tag] = dp[tag][0:cut_position]
    return data, targets


def read_data(bench=None):
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
    ind_train = indices[0:int(np.floor(0.5 * n_configs))]
    ind_val = indices[int(np.floor(0.5 * n_configs)):int(np.floor(0.75 * n_configs))]
    ind_test = indices[int(np.floor(0.75 * n_configs)):]

    array_data = np.array(data)
    train_data = array_data[ind_train]
    val_data = array_data[ind_val]
    test_data = array_data[ind_test]

    # Cut curves for validation and test
    cut_position = 11
    val_data, val_targets = cut_data(val_data, cut_position)
    test_data, test_targets = cut_data(test_data, cut_position)
    train_data, train_targets = cut_data(train_data, 51)  # Cut last value as it is repeated

    return train_data, val_data, test_data, train_targets, val_targets, test_targets


def get_data(train_data, val_data, test_data, train_targets, val_targets, test_targets):
    """
    Function returns training and validation data and labels normalized [0-1]
    """

    # lets see how the targets look likes
    # print(type(train_targets))
    # print(train_targets)
    # print(len(train_targets))

    # we have to predict the validation accuracy
    # in the data, for each config, we have 50 epochs, each epoch has a validation accuracy
    # use only 10 epochs for training and predict the 51st epoch val/accuracy

    # print(config_0["Train/val_accuracy"])
    # print(len(config_0["Train/val_accuracy"]))

    # ------------ new code - just make column vectors for each config, so (1000, 10, 1) is the shape of training data ---------------#
    # we will only use the config parmas and the train/val_accuracy, so 8 features per input
    train_data_new_X = np.zeros((1000, 8, 1))
    train_data_new_Y = np.zeros((1000, 1))

    val_data_X = np.zeros((500, 8, 1))
    val_data_Y = np.zeros((500, 1))

    test_data_X = np.zeros((500, 8, 1))
    test_data_Y = np.zeros((500, 1))

    for index, data in enumerate(train_data):
        # print(index, data)
        train_data_new_X[index, 0] = data["config"]["batch_size"]  # config params are single value scalars
        train_data_new_X[index, 1] = data["config"]["max_dropout"]
        train_data_new_X[index, 2] = data["config"]["max_units"]
        train_data_new_X[index, 3] = data["config"]["num_layers"]
        train_data_new_X[index, 4] = data["config"]["learning_rate"]
        train_data_new_X[index, 5] = data["config"]["momentum"]
        train_data_new_X[index, 6] = data["config"]["weight_decay"]

        # ------- dynamic params -----------#
        train_data_new_X[index, 7] = np.sum(
            np.asarray(data["Train/val_accuracy"][0:10])) / 10  # average train_val acc across 10 epochs

        # train_data_new_X[index, 7] = np.sum(np.asarray(data["Train/loss"][0:10])) / 10  # these we use only first 10 epochs, average loss
        # train_data_new_X[index, 8] = np.sum(np.asarray(data["Train/train_accuracy"][0:10])) / 10 # average accuracy across 10 epochs
        # train_data_new_X[index, 9] = np.sum(np.asarray(data["Train/val_accuracy"][0:10])) / 10 # average val_acc across 10 epochs
        # train_data_new_X[index, 10] = np.sum(np.asarray(data["Train/train_cross_entropy"][0:10])) / 10
        # train_data_new_X[index, 11] = np.sum(np.asarray(data["Train/val_cross_entropy"][0:10])) / 10
        # train_data_new_X[index, 12] = np.sum(np.asarray(data["Train/gradient_mean"][0:10])) / 10
        # train_data_new_X[index, 13] = np.sum(np.asarray(data["Train/lr"][0:10])) / 10

        train_data_new_Y[index] = train_targets[index]  # validation accuracy for 51st epoch, for each

        # print("target for config 0 : {0}, from dataset : {1}".format(train_data_new_Y[index], data["Train/val_accuracy"][50]))

        # print(train_data_new_X[0])

    # normalize each feature across all batches, so each feature is in range [0-1]
    for feat_index in range(8):
        feat_min = np.min(train_data_new_X[:, feat_index, 0])  # min of feature 0 across all batches
        feat_max = np.max(train_data_new_X[:, feat_index, 0])
        # print(feat_min, feat_max)

        # normalize input features
        train_data_new_X[:, feat_index, 0] = (train_data_new_X[:, feat_index, 0] - feat_min) / (feat_max - feat_min)
        # print(train_data_new_X[:, feat_index], train_data_new_X[:, feat_index].shape)

    # normalize labels
    feat_min = np.min(train_data_new_Y)
    feat_max = np.max(train_data_new_Y)

    train_data_new_Y = (train_data_new_Y - feat_min) / (feat_max - feat_min)  # normalize labels

    # ------- for validation set --------------#

    for index, data in enumerate(val_data):
        # print(index, data)
        val_data_X[index, 0] = data["config"]["batch_size"]  # config params are single value scalars
        val_data_X[index, 1] = data["config"]["max_dropout"]
        val_data_X[index, 2] = data["config"]["max_units"]
        val_data_X[index, 3] = data["config"]["num_layers"]
        val_data_X[index, 4] = data["config"]["learning_rate"]
        val_data_X[index, 5] = data["config"]["momentum"]
        val_data_X[index, 6] = data["config"]["weight_decay"]

        # ------- dynamic params -----------#
        val_data_X[index, 7] = np.sum(np.asarray(data["Train/val_accuracy"][0:10])) / 10

        # val_data_X[index, 7] = np.sum(np.asarray(data["Train/loss"][0:10])) / 10  # these we use only first 10 epochs, average loss
        # val_data_X[index, 8] = np.sum(np.asarray(data["Train/train_accuracy"][0:10])) / 10 # average accuracy across 10 epochs
        # val_data_X[index, 9] = np.sum(np.asarray(data["Train/val_accuracy"][0:10])) / 10 # average val_acc across 10 epochs

        # val_data_X[index, 7] = np.sum(np.asarray(data["Train/loss"][0:10])) / 10  # these we use only first 10 epochs, average loss
        # val_data_X[index, 8] = np.sum(np.asarray(data["Train/train_accuracy"][0:10])) / 10 # average accuracy across 10 epochs
        # val_data_X[index, 9] = np.sum(np.asarray(data["Train/val_accuracy"][0:10])) / 10 # average val_acc across 10 epochs
        # val_data_X[index, 10] = np.sum(np.asarray(data["Train/train_cross_entropy"][0:10])) / 10
        # val_data_X[index, 11] = np.sum(np.asarray(data["Train/val_cross_entropy"][0:10])) / 10
        # val_data_X[index, 12] = np.sum(np.asarray(data["Train/gradient_mean"][0:10])) / 10
        # val_data_X[index, 13] = np.sum(np.asarray(data["Train/lr"][0:10])) / 10

        val_data_Y[index] = val_targets[index]  # validation accuracy for 51st epoch, for each

    # normalize each feature across all batches, so each feature is in range [0-1]
    for feat_index in range(8):
        feat_min = np.min(val_data_X[:, feat_index, 0])  # min of feature 0 across all batches
        feat_max = np.max(val_data_X[:, feat_index, 0])
        print(feat_min, feat_max)

        # normalize input features
        val_data_X[:, feat_index, 0] = (val_data_X[:, feat_index, 0] - feat_min) / (feat_max - feat_min)
        # print(val_data_X[:, feat_index], val_data_X[:, feat_index].shape)

    # normalize labels
    feat_min = np.min(val_data_Y)
    feat_max = np.max(val_data_Y)

    val_data_Y = (val_data_Y - feat_min) / (feat_max - feat_min)  # normalize labels

    # ------- for test set --------------#

    for index, data in enumerate(test_data):
        # print(index, data)
        test_data_X[index, 0] = data["config"]["batch_size"]  # config params are single value scalars
        test_data_X[index, 1] = data["config"]["max_dropout"]
        test_data_X[index, 2] = data["config"]["max_units"]
        test_data_X[index, 3] = data["config"]["num_layers"]
        test_data_X[index, 4] = data["config"]["learning_rate"]
        test_data_X[index, 5] = data["config"]["momentum"]
        test_data_X[index, 6] = data["config"]["weight_decay"]

        # ------- dynamic params -----------#
        test_data_X[index, 7] = np.sum(np.asarray(data["Train/val_accuracy"][0:10])) / 10

        # test_data_X[index, 7] = np.sum(np.asarray(data["Train/loss"][0:10])) / 10  # these we use only first 10 epochs, average loss
        # test_data_X[index, 8] = np.sum(np.asarray(data["Train/train_accuracy"][0:10])) / 10 # average accuracy across 10 epochs
        # test_data_X[index, 9] = np.sum(np.asarray(data["Train/val_accuracy"][0:10])) / 10 # average val_acc across 10 epochs
        # test_data_X[index, 10] = np.sum(np.asarray(data["Train/train_cross_entropy"][0:10])) / 10
        # test_data_X[index, 11] = np.sum(np.asarray(data["Train/val_cross_entropy"][0:10])) / 10
        # test_data_X[index, 12] = np.sum(np.asarray(data["Train/gradient_mean"][0:10])) / 10
        # test_data_X[index, 13] = np.sum(np.asarray(data["Train/lr"][0:10])) / 10

        test_data_Y[index] = test_targets[index]  # validation accuracy for 51st epoch, for each

    # normalize each feature across all batches, so each feature is in range [0-1]
    for feat_index in range(8):
        feat_min = np.min(test_data_X[:, feat_index, 0])  # min of feature 0 across all batches
        feat_max = np.max(test_data_X[:, feat_index, 0])
        print(feat_min, feat_max)

        # normalize input features
        test_data_X[:, feat_index, 0] = (test_data_X[:, feat_index, 0] - feat_min) / (feat_max - feat_min)
        # print(val_data_X[:, feat_index], val_data_X[:, feat_index].shape)

    # normalize labels
    feat_min = np.min(test_data_Y)
    feat_max = np.max(test_data_Y)

    test_data_Y = (test_data_Y - feat_min) / (feat_max - feat_min)  # normalize labels

    # print(val_data_Y)

    return train_data_new_X, train_data_new_Y, val_data_X, val_data_Y, test_data_X, test_data_Y


# build the model here
class LCPredict(nn.Module):

    def __init__(self):
        super(LCPredict, self).__init__()

        self.lin1 = nn.Linear(8, 16)
        self.sig1 = nn.ReLU()
        self.drp1 = nn.Dropout(p=0.3)  # probability of being zeroed

        self.lin2 = nn.Linear(16, 8)
        self.sig2 = nn.ReLU()
        self.drp2 = nn.Dropout(p=0.2)

        # no dropout for last block
        self.lin3 = nn.Linear(8, 1)
        self.sig3 = nn.ReLU()

    def forward(self, x):
        x = self.sig1(self.lin1(x))
        x = self.drp1(x)

        x = self.sig2(self.lin2(x))
        x = self.drp2(x)

        x = self.lin3(x)

        return x


def train_model(model, loss, optimizer, train_X, train_Y, b_size, epochs):
    """
    Function trains the model

    Returns trained model
    """
    loss_list = []  # list to hold losses over epochs
    print(summary(model, (16, 8)))
    n_full_batches = int(train_X.shape[0] / b_size)  # number of full bacthes of b_size possible
    n_last_batch = train_X.shape[0] - (n_full_batches * b_size)  # size of last batch

    print("Number of full batches : {0}, each batch : {1} samples".format(n_full_batches, b_size))
    print("Last batch size : {0}".format(n_last_batch))

    total_loss = 0.0
    for n_epoch in range(epochs):
        print("Epoch {0}.......".format(n_epoch))
        print("Total data samples : {0}".format(train_X.shape[0]))
        start_index = 0
        for b in range(n_full_batches):
            optimizer.zero_grad()  # zero out the gradients
            outputs = model(train_X[start_index: start_index + b_size].squeeze_(-1)) # the squeeze is needed to set the dimensions right
            loss = loss_fn(outputs, train_Y[start_index: start_index + b_size])  # compute loss
            loss.backward() # backprop and gradients
            optimizer.step() # weight update

            print("data[{0} : {1}] ----> loss : {2}".format(start_index, start_index + b_size, loss.item()))

            # change the start index
            start_index += b_size

            # print(outputs.shape)
        # last batch has a different shape
        if n_last_batch != 0:
            optimizer.zero_grad()
            outputs = model(train_X[start_index: start_index + n_last_batch].squeeze_(-1))  # the squeeze is needed to set the dimensions right
            loss = loss_fn(outputs, train_Y[start_index: start_index + n_last_batch])  # compute loss
            loss.backward()  # backprop and gradients
            optimizer.step()  # weight update

            print("data[{0} : {1}] ----> loss : {2}".format(start_index, start_index + n_last_batch, loss.item()))

        loss_list.append(loss.item())  # store the last loss for this epoch

    print(loss_list, len(loss_list))
    return model, loss_list


def test_model(model, test_X, test_Y):
    """
    Function does prediction using the trained model and returns the results of prediction and errors
    :param model: trained model
    :param test_X: input data
    :param test_Y: true value to compare prediction against for statistics
    :return: predictions, error for each prediction, average error across all predictions, max error in all predictions, min error in all predictions
    """

    predictions = model(test_X.squeeze_(-1))

    error_abs = []
    error_signed = []
    error_mse = 0
    error_mean = 0
    error_max = 0
    error_min = 0

    print("True\t\t\tPredicted\t\t\terror(%)")
    for i, val in enumerate(test_Y):
        t_val = val.item()
        pred = predictions[i].item()

        # get absolute error for each prediction
        error_abs.append(np.abs(pred - t_val))
        error_signed.append(pred - t_val)

        print("{0}\t\t{1}\t\t{2}".format(t_val, pred, (np.abs(pred-t_val)/(t_val+0.00001))*100))

    # get mean error
    error_mean = np.mean(np.array(error_abs))

    # get mse error
    error_mse = np.square(error_mean)
    error_max = np.max(np.array(error_abs))  # get the maximum error
    error_min = np.min(np.array(error_abs))  # get minimum error

    return predictions, error_abs, error_signed, error_mean, error_mse, error_max, error_min





if __name__ == "__main__":
    print("------------- Let us predict some learning curves --------------")

    data_path = '/home/sambit/PROGRAMMING/DL_PROJECT/TEAM_WORK_FREIBURG/Extrapolation-of-Learning-Curves/DATA/fashion_mnist.json'
    data_root = Benchmark(data_dir=data_path)

    train_data, val_data, test_data, train_targets, val_targets, test_targets = read_data(data_root)

    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

    train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(train_data, val_data, test_data, train_targets,
                                                              val_targets, test_targets)  # get the prepared data

    print(train_X.shape, train_Y.shape, val_X.shape, val_Y.shape)

    # get model and send to GPU
    # model = LCPredict()
    device = torch.device("cuda")
    # model.to(device=device)

    # convert all data to torch tensors and send to GPU
    train_X_tensor = torch.from_numpy(train_X).float().to(device=device)
    train_Y_tensor = torch.from_numpy(train_Y).float().to(device=device)
    val_X_tensor = torch.from_numpy(val_X).float().to(device=device)
    val_Y_tensor = torch.from_numpy(val_Y).float().to(device=device)
    test_X_tensor = torch.from_numpy(test_X).float().to(device=device)
    test_Y_tensor = torch.from_numpy(test_Y).float().to(device=device)

    # create a loss fuction
    loss_fn = nn.MSELoss()

    # to be used in training 5 models
    lr_list = [0.002, 0.002, 0.002, 0.002, 0.002]
    b_size_list = [8, 8, 8, 8, 8]
    epochs_list = [100, 100, 100, 100, 100]

    mean_error_models = []
    mse_error_models = []

    for i in range(5):
        lr = lr_list[i]
        b_size = b_size_list[i]
        epochs = epochs_list[i]

        # lr = 0.002
        # b_size = 16
        # epochs = 200

        # get model and send to GPU, we need a new model each time since otherwise we would be starting from the previous model
        model = LCPredict()
        # device = torch.device("cuda")
        model.to(device=device)


        # write to the file
        fd = open("model_"+str(i)+'.txt', 'w')
        # write the model to the file
        fd.write(str(model))

        # create an optimizer
        optimizer_adam = Adam(model.parameters(), lr=lr)
        fd.write("\noptimizer : adam, lr : "+str(lr))
        fd.write("\nepochs : "+str(epochs))
        fd.write("\nbatch size : "+str(b_size))

        # training loop
        model, loss_list_epoch = train_model(model, loss_fn, optimizer_adam, train_X_tensor, train_Y_tensor, b_size, epochs)

        fd.write("\nloss : "+str(loss_list_epoch))

        # save the model
        torch.save(model, "full_model_"+str(i)+'.pth')

        x_val = np.arange(0, len(loss_list_epoch))
        plt.plot(x_val, loss_list_epoch)
        plt.grid(color='r', linewidth=1)
        plt.show(block=False)
        plt.pause(3)
        # plt.close()
        plt.savefig("loss_model_"+str(i)+'.pdf')
        plt.close()


        predictions, error_abs, error_signed, error_mean, error_mse, error_max, error_min = \
            test_model(model, test_X_tensor, test_Y_tensor)

        x_val = np.arange(0, 500)

        # plot and save the errors
        error_abs_arr = np.array(error_abs)
        plt.plot(x_val, error_abs_arr*100)
        plt.grid(color='r', linewidth=1)
        plt.show(block=False)
        plt.pause(3)
        # plt.close()
        plt.savefig("error_abs_model_"+str(i)+'.pdf')
        plt.close()

        error_signed_arr = np.array(error_signed)
        plt.plot(x_val, error_signed_arr*100)
        plt.grid(color='r', linewidth=1)
        plt.show(block=False)
        plt.pause(3)
        # plt.close()
        plt.savefig("error_signed_model_" + str(i) + '.pdf')
        plt.close()

        # write to file
        fd.write("\nerror_abs : "+str(error_abs))
        fd.write("\nerror_signed : "+str(error_signed))
        fd.write("\nerror_mean : "+str(error_mean*100))
        fd.write("\nerror_mse : "+str(error_mse*100))
        fd.write("\nerror_max : "+str(error_max*100))
        fd.write("\nerror_min : "+str(error_min*100))

        mean_error_models.append(error_mean * 100)
        mse_error_models.append(error_mse * 100)

    x_val = np.arange(0, 5)
    plt.plot(x_val, mean_error_models)
    plt.grid(color='r', linewidth=1)
    plt.show(block=False)
    plt.pause(3)
    plt.savefig("mean_error_5_models.pdf")
    plt.close()

    x_val = np.arange(0, 5)
    plt.plot(x_val, mse_error_models)
    plt.grid(color='r', linewidth=1)
    plt.show(block=False)
    plt.pause(3)
    plt.savefig("mse_error_5_models.pdf")
    plt.close()





