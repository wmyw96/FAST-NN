import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class FactorAugmentedNN(nn.Module):
	def __init__(self, p, r_bar, depth, width, dp_mat):
		super(FactorAugmentedNN, self).__init__()

		self.diversified_projection = nn.Linear(p, r_bar, bias=False)
		dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32)
		self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=False)

		relu_nn = [('linear1', nn.Linear(r_bar, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x):
		x = self.diversified_projection(x)
		pred = self.relu_stack(x)
		return pred


class RegressionNN(nn.Module):
	def __init__(self, d, depth, width):
		super(RegressionNN, self).__init__()
		
		relu_nn = [('linear1', nn.Linear(d, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))
		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))

		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x):
		pred = self.relu_stack(x)
		return pred


