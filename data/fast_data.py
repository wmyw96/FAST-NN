import numpy as np
from torch.utils.data import Dataset
import torch
import data.univariate_funcs as univariate_funcs


class AdditiveModel:

	def __init__(self, num_funcs, rd_size=5, normalize=True):
		self.func_zoo = [
			univariate_funcs.func1,
			univariate_funcs.func2,
			univariate_funcs.func3,
			univariate_funcs.func4,
			univariate_funcs.func5,
			univariate_funcs.func6,
			univariate_funcs.func7,
			univariate_funcs.func8,
			univariate_funcs.func9
		]
		self.func_name = [
			'sin',
			'sqrt_abs',
			'exp',
			'sigmoid',
			'cos_pi',
			'sin_pi',
			'-sin',
			'cos_2',
			'tan'
		]
		self.num_funcs = num_funcs
		self.func_idx = np.random.randint(0, rd_size, num_funcs)
		self.func_idx[0] = 4
		self.func_idx[4] = 4
		self.normalize = normalize

	def sample(self, x):
		y = np.zeros((np.shape(x)[0], 1))
		if np.shape(x)[1] != self.num_funcs:
			raise ValueError("AdditiveModel: Data dimension {}, ".format(np.shape(0)) +
							 "number of additive functions = {}".format(self.num_funcs))
		for i in range(self.num_funcs):
			y = y + self.func_zoo[self.func_idx[i]](x[:, i:i + 1])
		if self.normalize:
			y = y / self.num_funcs
		return y

	def __str__(self):
		s = "Additive Models: f(x) = \n"
		for i in range(self.num_funcs):
			s = s + f"      {self.func_name[self.func_idx[i]]} (x_{i + 1})\n"
		return s


class HierarchicalCompositionModels:
	def __init__(self, idx):
		self.func_idx = idx

	def sample(self, x):
		y = np.zeros((np.shape(x)[0], 1))
		if self.func_idx == 0:
			for i in range(np.shape(x)[1]):
				#print(np.shape(y))
				y = y + ((-1) ** (i+1)) * x[:, i:i+1]
				#print(np.shape(x[:, i]))
				#print(np.shape(y))
			print(np.shape(y))
		if self.func_idx == 1:
			y = np.sin(np.sum(np.abs(x), 1))
		if self.func_idx == 2:
			y = np.exp(np.sqrt(np.sum(np.square(x), 1))) * (x[:, 1] >= x[:, 2]**2) + \
				   x[:, 2] - x[:, 4] * x[:, 5] * (x[:, 6]**4)
		if self.func_idx == 3:
			y = x[:, 0] * (x[:, 1]**2) - x[:, 2] + \
				   np.log(x[:, 3] + 4 * x[:, 4] + 8 + np.exp(x[:, 5] * x[:, 6] - 5 * x[:, 4])) + \
				np.tan(x[:, 7] + 0.1) + np.sin(x[:, 8])
		if self.func_idx == 5:
			y = 4 * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * (x[:, 3] + x[:, 2] >= 1) * (x[:, 4] >= x[:, 5]) + \
				np.cos(np.sum(np.abs(x), 1)) * ((x[:, 0] ** 2) * x[:, 6] * x[:, 7] >= x[:, 8] * (x[:, 5] ** 3))
		y = np.reshape(y, (np.shape(y)[0], 1))
		return y


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
