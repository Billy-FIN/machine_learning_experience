import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    with open('datasets/train_data_x.pkl','rb') as f:
        train_data_x = pkl.load(f)

    with open('datasets/train_data_y.pkl','rb') as f:
        train_data_y = pkl.load(f)

    with open('datasets/test_data_x.pkl','rb') as f:
        test_data_x = pkl.load(f)

    with open('datasets/test_data_y.pkl','rb') as f:
        test_data_y = pkl.load(f)

    return train_data_x, train_data_y, test_data_x, test_data_y

# CNN model
class Net(nn.Module):










