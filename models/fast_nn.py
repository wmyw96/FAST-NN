import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class FactorAugmentedSparseThroughputNN(nn.Module):
	def __init__(self, p, r_bar, depth, width, dp_mat):
		super(FactorAugmentedSparseThroughputNN, self).__init__()

		self.diversified_projection = nn.Linear(p, r_bar, bias=False)
		dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32)
		self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=False)

		self.variable_selection = nn.Linear(p, width, bias=False)
		relu_nn = [('linear1', nn.Linear(width + r_bar, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x):
		x1 = self.diversified_projection(x)
		x2 = self.variable_selection(x)
		pred = self.relu_stack(torch.concat((x1, x2), -1))
		return pred

	def regularization_loss(self, tau):
		clipped_l1 = torch.clamp(torch.abs(self.variable_selection.weight) / tau, max=1.0)
		return torch.sum(clipped_l1)

