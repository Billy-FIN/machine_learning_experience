import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import cnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

# model train
def model_train(train_data_x, train_data_y, test_data_x, test_data_y):



# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, epoch_num):



if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# rescale data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0


	# model train (model test function can be called directly in model_train)
	model_train(train_data_x, train_data_y, test_data_x, test_data_y)










