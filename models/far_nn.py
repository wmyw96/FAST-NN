import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class FactorAugmentedNN(nn.Module):
	def __init__(self, p, r_bar, depth, width, dp_mat, fix_dp_mat=True, input_dropout=False,
				 with_x=False, dropout_rate=0.0):
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
		if self.use_input_dropout and is_training:
			x = self.input_dropout(x)
		ft = self.diversified_projection(x)
		if self.with_x:
			pred = self.relu_stack(torch.concat([ft, x], 1))
		else:
			pred = self.relu_stack(ft)
		return pred


class RegressionNN(nn.Module):
	def __init__(self, d, depth, width, input_dropout=False, dropout_rate=0.0):
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
		if self.use_input_dropout and is_training:
			x = self.input_dropout(x)
		pred = self.relu_stack(x)
		return pred


