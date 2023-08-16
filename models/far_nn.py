import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class FactorAugmentedNN(nn.Module):
	'''
		A class used to implement FAR-NN

		...
		Attributes
		----------
		use_input_dropout : bool
			whether to use dropout (True) or not (False) in the input layer
		input_dropout : nn.module
			pytorch module of input dropout
		diversitied_projection : nn.module
			the implementation of diversified projection matrix
		with_x : bool
			whether to augment x as input (True) or not (False)
		relu_stack : nn.module
			pytorch module to implement relu neural network 

		Methods
		----------
		__init__(x, is_training=False)
			Initialize the module
		forward()
			Implementation of forwards pass
	'''
	def __init__(self, p, r_bar, depth, width, dp_mat, fix_dp_mat=True, input_dropout=False,
				 with_x=False, dropout_rate=0.0):
		'''
			Parameters
			----------
			p : int
				input dimension
			r_bar : int
				the hyper-parameter r bar in the main paper
			depth : int
				the number of hidden layers of relu network
			width : int
				the number of units in each hidden layer of relu network
			dp_mat : numpy.array
				the fixed pretrained (p, r_bar) diversfied projection matrix 
			fix_dp_mat : bool, optional
				whether to use fixed pretrained diversfied projection matrix (True)
			input_dropout : bool, optional
				whether to use input dropout in the input layer (True)
			with_x : bool, optional
				whether to augment x as input (True)
			dropout_rate: float, optional
				the dropout rate for the input dropout
		'''
		super(FactorAugmentedNN, self).__init__()

		self.use_input_dropout = input_dropout
		self.input_dropout = nn.Dropout(p=dropout_rate)
		self.diversified_projection = nn.Linear(p, r_bar, bias=False)
		dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32)
		self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=not fix_dp_mat)
		self.with_x = with_x

		if with_x:
			relu_nn = [('linear1', nn.Linear(r_bar + p, width)), ('relu1', nn.ReLU())]
		else:
			relu_nn = [('linear1', nn.Linear(r_bar, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x, is_training=False):
		'''
			Parameters
			----------
			x : torch.tensor
				the (n x p) matrix of the input
			is_training : bool
				whether the forward pass is used in the training (True) or not,
				used for dropout module

			Returns
			----------
			pred : torch.tensor
				(n, 1) matrix of the prediction
		'''
		if self.use_input_dropout and is_training:
			x = self.input_dropout(x)
		ft = self.diversified_projection(x)
		if self.with_x:
			pred = self.relu_stack(torch.concat([ft, x], 1))
		else:
			pred = self.relu_stack(ft)
		return pred


class RegressionNN(nn.Module):
	'''
		A class to implement standard relu nn for regression

		...
		Attributes
		----------
		use_input_dropout : bool
			whether to use dropout (True) or not (False) in the input layer
		input_dropout : nn.module
			pytorch module of input dropout
		relu_stack : nn.module
			pytorch module to implement relu neural network

		Methods
		----------
		__init__(x, is_training=False)
			Initialize the module
		forward()
			Implementation of forwards pass
	'''
	def __init__(self, d, depth, width, input_dropout=False, dropout_rate=0.0):
		'''
			Parameters
			----------
			d : int
				input dimension
			depth : int
				the number of hidden layers of relu network
			width : int
				the number of units in each hidden layer of relu network
			input_dropout : bool, optional
				whether to use input dropout in the input layer (True)
			dropout_rate: float, optional
				the dropout rate for the input dropout
		'''
		super(RegressionNN, self).__init__()
		self.use_input_dropout = input_dropout
		self.input_dropout = nn.Dropout(p=dropout_rate)

		relu_nn = [('linear1', nn.Linear(d, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))
		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x, is_training=False):
		'''
			Parameters
			----------
			x : torch.tensor
				the (n x p) matrix of the input
			is_training : bool
				whether the forward pass is used in the training (True) or not,
				used for dropout module

			Returns
			----------
			pred : torch.tensor
				(n, 1) matrix of the prediction
		'''
		if self.use_input_dropout and is_training:
			x = self.input_dropout(x)
		pred = self.relu_stack(x)
		return pred


