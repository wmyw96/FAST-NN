import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class FactorAugmentedSparseThroughputNN(nn.Module):
	'''
		A class to implement the FAST-NN 

		...
		Attributes
		----------
		diversified_projection : nn.module
			implementation of diversified projection matrix
		reconstruct : nn.module
			a linear module to estimate (covariate - idiosyncratic error)
		variable_selection : nn.module
			implementation of variable selection matrix
		relu_stack: nn.module
			the relu neural network module

		Methods
		----------
		__init__()
			Initialize the module
		forward(x, is_traininig=False)
			Implementation of forwards pass
		regularization_loss(tau, penalize_weights)
			Calculate the regularization term for the variable selection matrix
	'''
	def __init__(self, p, r_bar, depth, width, dp_mat, sparsity=None, rs_mat=None):
		'''
			Parameters
			----------
			p : int
				input dimension
			r_bar : r_bar
				the number of diversified projection weights
			depth : int
				the number of hidden layers of neural network
			width : int
				the number of hidden units in each layer
			sparsity : int
				the variable selection matrix is (p, sparsity) matrix
			rs_mat : np.array
				(r_bar, p) reconstruction matrix 
		'''
		super(FactorAugmentedSparseThroughputNN, self).__init__()

		self.diversified_projection = nn.Linear(p, r_bar, bias=False)
		dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32)
		self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=False)

		if rs_mat is not None:
			self.reconstruct = nn.Linear(r_bar, p, bias=False)
			rs_matrix_tensor = torch.tensor(np.transpose(rs_mat), dtype=torch.float32)
			self.reconstruct.weight = nn.Parameter(rs_matrix_tensor, requires_grad=False)
		else:
			self.reconstruct = None

		if sparsity is None:
			sparsity = width
		self.variable_selection = nn.Linear(p, sparsity, bias=False)

		relu_nn = [('linear1', nn.Linear(r_bar + sparsity, width)), ('relu1', nn.ReLU())]
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
				(n, p) matrix of the input
			is_training : bool
				whether the forward pass is used in the training (True) or not

			Returns
			----------
			pred : torch.tensor
				(n, 1) matrix of the prediction
		'''
		x1 = self.diversified_projection(x)
		if self.reconstruct is not None:
			x2 = self.variable_selection(x - self.reconstruct(x1))
		else:
			x2 = self.variable_selection(x)
		pred = self.relu_stack(torch.concat((x1, x2), -1))
		return pred

	def regularization_loss(self, tau, penalize_weights=False):
		'''
			Parameters
			----------
			tau : float
				the hyper-parameter tau in the paper
			penalize_weights : bool
				whether to apply the L1 regularization to the neural network weights

			Returns
			----------
			value : torch.tensor
				a scalar of the regularization loss
		'''
		l1_penalty = torch.abs(self.variable_selection.weight) / tau
		clipped_l1 = torch.clamp(l1_penalty, max=1.0)
		# input_l1_norm = torch.sum(clipped_l1, 1)   # shape = [width,]
		if penalize_weights:
			for param in self.relu_stack.parameters():
				if len(param.shape) > 1:
					clipped_l1 += 0.001 * torch.sum(torch.abs(param))
		return torch.sum(clipped_l1)
