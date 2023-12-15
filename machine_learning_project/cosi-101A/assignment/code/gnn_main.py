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
	# model definition
	loss_fn = nn.CrossEntropyLoss()
	model = U.GCN(nfeat=features.shape[1]).to("cuda")
	optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	
	model.train()
	for epoch in range(200):
		outputs = model(features, adj)
		loss_train = loss_fn(outputs[idx_train], labels[idx_train])

		# backward and optimize
		optimizer.zero_grad()
		loss_train.backward()
		optimizer.step()

		print('Epoch: {}'.format(epoch+1),
			  'loss_train: {:.4f}'.format(loss_train.item()))
		
	model_test(model, adj, features, labels, idx_test)


# model test: can be called directly in model_train 
def model_test(model, adj, features, labels, idx_test, loss_fn=nn.CrossEntropyLoss()):
	model.eval()
	outputs = model(features, adj)

	# loss and accuracy
	loss_test = loss_fn(outputs[idx_test], labels[idx_test])
	acc_test = U.accuracy(outputs[idx_test], labels[idx_test])
	
	print("Test set results:",
		  "loss= {}".format(loss_test.item()),
		  "accuracy= {:.4f}".format(acc_test.item()))

if __name__ == '__main__':
	# load datasets
	adj, features, labels, idx_train, idx_test = U.load_data()

	# move to cuda
	adj = adj.to("cuda")
	features = features.to("cuda")
	labels = labels.to("cuda")
	idx_train = idx_train.to("cuda")
	idx_test = idx_test.to("cuda")

	# model train (model test function can be called directly in model_train)
	model_train(adj, features, labels, idx_train, idx_test)






