import numpy as np
from torch.utils.data import Dataset
import torch
import data.univariate_funcs as univariate_funcs


class AdditiveModel:

	def __init__(self, num_funcs, normalize=True):
		self.func_zoo = [
			univariate_funcs.func1,
			univariate_funcs.func2,
			univariate_funcs.func3,
			univariate_funcs.func4,
			univariate_funcs.func5
		]
		self.func_name = [
			'sin',
			'sqrt_abs',
			'exp',
			'sigmoid',
			'cos_pi'
		]
		self.num_funcs = num_funcs
		self.func_idx = np.random.randint(0, 5, num_funcs)
		self.normalize = normalize

	def sample(self, x):
		y = np.zeros((np.shape(x)[0], 1))
		if np.shape(x)[1] != self.num_funcs:
			raise ValueError("AdditiveModel: Data dimension {}, ".format(np.shape(0)) +
							 "number of additive functions = {}".format(self.num_funcs))
		for i in range(self.num_funcs):
			y = y + self.func_zoo[self.func_idx[i]](x[:, i:i+1])
		if self.normalize:
			y = y / self.num_funcs
		return y

	def __str__(self):
		s = "Additive Models: f(x) = \n"
		for i in range(self.num_funcs):
			s = s + f"      {self.func_name[self.func_idx[i]]} (x_{i+1})\n"
		return s


class RegressionDataset(Dataset):
	def __init__(self, x, y):
		self.n = np.shape(x)[0]
		if self.n != np.shape(y)[0]:
			raise ValueError("RegressionDataset: Sample size doesn't match!")
		self.feature = x
		self.response = y

	def __len__(self):
		return self.n

	def __getitem__(self, idx):
		return torch.tensor(self.feature[idx, :], dtype=torch.float32), \
					torch.tensor(self.response[idx, :], dtype=torch.float32)
