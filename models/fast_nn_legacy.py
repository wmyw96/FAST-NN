import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class FactorAugmentedSparseThroughputNN(nn.Module):
	def __init__(self, p, r_bar, depth, width, dp_mat, rs_mat=None):
		super(FactorAugmentedSparseThroughputNN, self).__init__()

		self.diversified_projection = nn.Linear(p, r_bar, bias=False)
		dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32)
		self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=False)

		if rs_mat is not None:
			print('Reconstruction Part')
			self.reconstruct = nn.Linear(r_bar, p, bias=False)
			rs_matrix_tensor = torch.tensor(np.transpose(rs_mat), dtype=torch.float32)
			self.reconstruct.weight = nn.Parameter(rs_matrix_tensor, requires_grad=False)
		else:
			self.reconstruct = None

		# legacy
		# self.variable_selection = nn.Linear(p, width, bias=False)

		self.variable_selection_logits = nn.Parameter(torch.empty((p, r_bar)))
		self.scale = nn.Parameter(torch.empty((1, r_bar)))
		with torch.no_grad():
			self.scale.uniform_(-1, 1)
			self.variable_selection_logits.uniform_(-1, 1)
			print(self.scale.detach().numpy())

		relu_nn = [('linear1', nn.Linear(r_bar + r_bar, width)), ('relu1', nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

		relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)

	def forward(self, x, is_training=False):
		x1 = self.diversified_projection(x)
		if self.reconstruct is not None:
			x2 = torch.matmul(x - self.reconstruct(x1),
							  torch.nn.functional.softmax(self.variable_selection_logits, dim=1))
			#x2 = x2 * self.scale
		else:
			x2 = torch.matmul(x,
							  torch.nn.functional.softmax(self.variable_selection_logits, dim=1))
			#x2 = x2 * self.scale
		pred = self.relu_stack(torch.concat((x1, x2), -1))
		return pred

	def regularization_loss(self, tau):
		l1_penalty = torch.abs(self.scale) / tau
		clipped_l1 = torch.clamp(l1_penalty, max=1.0)
		# input_l1_norm = torch.sum(clipped_l1, 1)   # shape = [width,]
		return torch.sum(clipped_l1)
