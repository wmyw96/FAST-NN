import numpy as np
from numpy import genfromtxt


class fred_md_data:
	def __init__(self, file_name='transfromed_data.csv', pred_index=None):
		self.data = genfromtxt(file_name, delimiter=',')
		self.pred_index = pred_index
		if pred_index is not None:
			n = np.shape(self.data)[0]
			covariate = self.data[0:n-1, :]
			response = self.data[1:, pred_index:pred_index+1]
			self.data = np.concatenate([response, covariate], 1)

		nan_indicator = (np.isnan(self.data))
		self.valid_rows = (np.sum(nan_indicator, axis=1, keepdims=True) == 0)
		cnt = 0
		for i in range(np.shape(self.data)[0]):
			if self.valid_rows[i]:
				cnt += 1
				if cnt == 200:
					print(i)
		self.n = np.shape(self.data)[0]
		self.valid_n = np.sum(self.valid_rows)
		print(f"(FRED-MD dataset): number of data = {self.n}, number of valid data = {np.sum(self.valid_rows)}")
		valid_data = []
		for i in range(self.n):
			if self.valid_rows[i]:
				valid_data.append(self.data[i, :])
		self.valid_data = np.array(valid_data)

		# print summary statistics
		# print('(FRED-MD dataset) Mean Stat:')
		# print(np.mean(self.valid_data, 0))
		# print('(FRED-MD dataset) Std Stat:')
		# print(np.std(self.valid_data, 0))

	def get_data(self, train_idx, test_idx, normalize=True):
		train_data = self.valid_data[train_idx, :]
		test_data = self.valid_data[test_idx, :]
		if normalize:
			mn_stat = np.mean(train_data, 0, keepdims=True)
			std_stat = np.std(train_data, 0, keepdims=True)
			train_data = (train_data - mn_stat) / std_stat
			test_data = (test_data - mn_stat) / std_stat
			return train_data, test_data, mn_stat, std_stat

	def get_split_data(self, train_idx, test_idx, split_ratio=0.7, normalize=True):
		train_idx_np = np.array(train_idx)
		np.random.shuffle(train_idx_np)
		n_split = int(len(train_idx) * split_ratio)
		train_data = self.valid_data[train_idx_np[:n_split], :]
		train_data2 = self.valid_data[train_idx_np[n_split:], :]
		test_data = self.valid_data[test_idx, :]
		if normalize:
			mn_stat = np.mean(train_data, 0, keepdims=True)
			std_stat = np.std(train_data, 0, keepdims=True)
			train_data = (train_data - mn_stat) / std_stat
			train_data2 = (train_data2 - mn_stat) / std_stat
			test_data = (test_data - mn_stat) / std_stat
			return train_data, train_data2, test_data, mn_stat, std_stat
		else:
			return train_data, train_data2, test_data



