import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import cnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import datetime
torch.manual_seed(0)

# model train
def model_train(train_data_x, train_data_y, test_data_x, test_data_y):
	
	# initialize the model
	net = U.Net()
	
	# define necessary parameters
	net = net.to(device="cuda")
	optimizer = optim.Adam(net.parameters(), lr=1e-4)
	loss_fn = nn.CrossEntropyLoss()
	batch_size = 5

	# convert numpy array to tensor
	train_data_x = torch.from_numpy(train_data_x).float().permute(0, 3, 1, 2).to(device="cuda")		# permute: change the order of dimensions that pytorch can recognize
	train_data_y = torch.from_numpy(train_data_y).long().to(device="cuda")

	# convert tensor to dataset
	train_data_y = train_data_y.view(-1)
	train_dataset = TensorDataset(train_data_x, train_data_y)

	# convert dataset to dataloader
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	# model train
	net.train()
	for epoch in range(100):
		loss_train = 0.0
		for i, data in enumerate(train_loader, 0):
			# get the inputs
			inputs, labels = data
			outputs = net(inputs)
			loss = loss_fn(outputs, labels)

			# backward + optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			loss_train += loss.item()

		if epoch == 1 or epoch % 10 == 0:
			print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
			
	model_test(test_data_x, test_data_y, net, batch_size)
	

# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, batch_size):
	# convert numpy array to tensor
	test_data_x = torch.from_numpy(test_data_x).float().permute(0, 3, 1, 2).to(device="cuda")
	test_data_y = torch.from_numpy(test_data_y).long().to(device="cuda")

	# convert tensor to dataset
	test_data_y = test_data_y.view(-1)
	test_dataset = TensorDataset(test_data_x, test_data_y)

	# convert dataset to dataloader
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	# model test
	correct = 0
	total = 0
	net.eval()
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.shape[0]
			correct += int((predicted == labels).sum())

	print('Accuracy of the network on test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# rescale data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0

	# model train (model test function can be called directly in model_train)
	model_train(train_data_x, train_data_y, test_data_x, test_data_y)










