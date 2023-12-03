import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import gnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

# model train
def model_train(adj, features, labels, idx_train, idx_test):


# model test: can be called directly in model_train 
def model_test(model, adj, features, labels, idx_test):


if __name__ == '__main__':
	# load datasets
	adj, features, labels, idx_train, idx_test = U.load_data()

	# model train (model test function can be called directly in model_train)
	model_train(adj, features, labels, idx_train, idx_test)






