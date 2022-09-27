import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math


class ParallelLinear(nn.Module):
	__constants__ = ['in_features', 'out_features']
	in_features: int
	out_features: int
	parallel_num: int
	weight: torch.Tensor

	def __init__(self, parallel_num: int, in_features: int, out_features: int, bias: bool = True,
				 device=None, dtype=None) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(ParallelLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.num_parallel = parallel_num
		self.weight = nn.Parameter(torch.empty((parallel_num, in_features, out_features), **factory_kwargs))
		if bias:
			self.bias = nn.Parameter(torch.empty((parallel_num, 1, out_features), **factory_kwargs))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters(in_features, out_features)

	def reset_parameters(self, fan_in: int, fan_out: int) -> None:
		bound = math.sqrt(3) * math.sqrt(2) / math.sqrt(fan_in)
		with torch.no_grad():
			self.weight.uniform_(-bound, bound)
			if self.bias is not None:
				self.bias.uniform_(-bound, bound)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		# print(f"Parallel Linear: input shape = {input.shape}, weight shape = {self.weight.shape}")
		return torch.matmul(input, self.weight) + self.bias

	def extra_repr(self) -> str:
		return 'num_parallel={}, in_features={}, out_features={}, bias={}'.format(
			self.parallel_num, self.in_features, self.out_features, self.bias is not None
		)


class SparseNeuralAdditiveModels(nn.Module):
	def __init__(self, p, depth, width, output_trunc=1.0, constrained_r=5.0):
		super(SparseNeuralAdditiveModels, self).__init__()
		relu_nn = [('linear{}'.format(1), ParallelLinear(p, 1, width)), ('relu{}'.format(1), nn.ReLU())]
		for i in range(depth - 1):
			relu_nn.append(('linear{}'.format(i+2), ParallelLinear(p, width, width)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))
		relu_nn.append(('linear{}'.format(depth + 1), ParallelLinear(p, width, 1)))
		self.relu_stack = nn.Sequential(OrderedDict(relu_nn))
		self.output_trunc_act = nn.Tanh()
		self.output_trunc_level = output_trunc
		self.beta_logits = nn.Parameter(torch.empty((1, p)))
		self.constrained_r = constrained_r
		with torch.no_grad():
			self.beta_logits.uniform_(1, 1)
			print('BETA SOFTMAX {}'.format(torch.nn.functional.softmax(self.beta_logits, dim=1).detach().numpy()))

	def forward(self, x, anneal=1.0, is_training=False):
		# x.shape: [n, p]
		x = x[:, :, None]
		# x.shape: [n, p, 1]
		x = torch.transpose(x, 0, 1)
		# x.shape: [p, n, 1]
		x = self.relu_stack(x)
		# x.shape: [p, n, 1]
		x = self.output_trunc_level * self.output_trunc_act(x)
		x = torch.transpose(x, 0, 1)
		# x.shape: [n, p, 1]
		x = torch.squeeze(x)
		# x.shape: [n, p]
		x = x * torch.nn.functional.softmax(self.beta_logits * anneal, dim=1)
		return self.constrained_r * torch.sum(x, dim=1, keepdim=True)

	def regularization_loss(self):
		return torch.sum(torch.abs(torch.nn.functional.softmax(self.beta_logits, dim=1)))


class FactorAugmentedNeuralAdditiveModels(nn.Module):

	def __init__(self, p, r_bar, depth_f, width_f, depth_u, width_u, dp_mat):
		super(FactorAugmentedNeuralAdditiveModels, self).__init__()

		self.diversified_projection = nn.Linear(p, r_bar, bias=False)
		dp_matrix_tensor = torch.tensor(np.transpose(dp_mat), dtype=torch.float32)
		self.diversified_projection.weight = nn.Parameter(dp_matrix_tensor, requires_grad=False)

		self.matrix_v = nn.Linear(r_bar, p, bias=False)

		relu_nn = [('linear1', nn.Linear(width_f + r_bar, width_f)), ('relu1', nn.ReLU())]
		for i in range(depth_f - 1):
			relu_nn.append(('linear{}'.format(i+2), nn.Linear(width_f, width_f)))
			relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

		relu_nn.append(('linear{}'.format(depth_f+1), nn.Linear(width_f, 1)))
		self.relu_stack = nn.Sequential(
			OrderedDict(relu_nn)
		)
