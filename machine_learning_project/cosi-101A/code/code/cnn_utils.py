import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():

    with open('D:/study/machine_learning_experience/machine_learning_project/cosi-101A/code/code/datasets/train_data_x.pkl','rb') as f:
        train_data_x = pkl.load(f)

    with open('D:/study/machine_learning_experience/machine_learning_project/cosi-101A/code/code/datasets/train_data_y.pkl','rb') as f:
        train_data_y = pkl.load(f)

    with open('D:/study/machine_learning_experience/machine_learning_project/cosi-101A/code/code/datasets/test_data_x.pkl','rb') as f:
        test_data_x = pkl.load(f)

    with open('D:/study/machine_learning_experience/machine_learning_project/cosi-101A/code/code/datasets/test_data_y.pkl','rb') as f:
        test_data_y = pkl.load(f)

    return train_data_x, train_data_y, test_data_x, test_data_y

# CNN model
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # first convolutional layer
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)

        # second convolutional layer
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)

        # fully connected layer
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, 12 * 5 * 5)   # reshape
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out








