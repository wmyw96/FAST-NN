from sklearn import linear_model
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh


class Lasso:
	def __init__(self, fold_validation=5):
		self.model = None
		self.choice_lambda = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
		self.choice_large_lambda = [0.2, 0.5, 1]
		self.choice_small_lambda = [0.001, 0.0005, 0.0002, 0.0001]
		self.fold = fold_validation

	def kfold_fit(self, x, y, fit_intercept=False):
		self.model = None
		min_error = 1e9
		best_alpha = 0
		n = np.shape(x)[0]
		block_size = n // self.fold
		for alpha in self.choice_lambda:
			mse = 0.0
			for i in range(self.fold):
				train_x = np.concatenate([x[:block_size*i, :], x[block_size*(i+1):, :]], 0)
				train_y = np.concatenate([y[:block_size*i,], y[block_size*(i+1):,]], 0)
				test_x = x[block_size*i: block_size*(i+1), :]
				test_y = y[block_size*i: block_size*(i+1),]
				a = linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=100000)
				a.fit(train_x, train_y)
				pred = a.predict(test_x)
				assert pred.shape == test_y.shape
				mse += np.mean(np.square(pred - test_y))
			if mse < min_error:
				min_error = mse
				best_alpha = alpha
		self.model = linear_model.Lasso(alpha=best_alpha, fit_intercept=fit_intercept, max_iter=100000)
		self.model.fit(x, y)

	def model_fit(self, x, y, test_x, test_y, candidate_alpha, fit_intercept=False):
		min_error = 1e9
		best_alpha = 0
		for alpha in candidate_alpha:
			tmp = linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=100000)
			tmp.fit(x, y)
			pred = tmp.predict(test_x)
			assert pred.shape == test_y.shape
			mse = np.mean(np.square(pred - test_y))
			# print(f"mse for alpha {alpha}: {mse / self.fold}")
			if mse < min_error:
				min_error = mse
				best_alpha = alpha
		return best_alpha, min_error

	def fit(self, x, y, test_x, test_y, fit_intercept=False):
		self.model = None
		valid_error = 0.0
		alpha_1, valid_error = self.model_fit(x, y, test_x, test_y, self.choice_lambda, fit_intercept)
		if alpha_1 == 0.2:
			alpha_1, valid_error = self.model_fit(x, y, test_x, test_y, self.choice_large_lambda, fit_intercept)
		elif alpha_1 == 0.001:
			alpha_1, valid_error = self.model_fit(x, y, test_x, test_y, self.choice_small_lambda, fit_intercept)
		self.model = linear_model.Lasso(alpha=alpha_1, fit_intercept=fit_intercept, max_iter=100000)
		self.model.fit(x, y)
		print(f"(Lasso Estimator) best alpha = {alpha_1}, valid mse = {valid_error}")

	def predict(self, x):
		if self.model is not None:
			return self.model.predict(x)
		else:
			raise ValueError('Please first fit the model')

	def fit_and_predict(self, x, y, valid_x, valid_y, test_x):
		self.fit(x, y, valid_x, valid_y, fit_intercept=False)
		return self.predict(test_x)


def estimate_factor_structure_from_observation(x, loadings):
	cov_b = np.matmul(np.transpose(loadings), loadings)
	inv_cov_b = np.linalg.inv(cov_b)
	factor = np.matmul(np.matmul(x, loadings), inv_cov_b)
	idiosyncratic = x - np.matmul(factor, np.transpose(loadings))
	return factor, idiosyncratic


class PCR:
	def __init__(self):
		self.model = None
		self.pc_map = None
		self.loading = None

	def fit(self, x, y, test_x, test_y, fit_intercept=False):
		# x: [n, p]
		# y: [n,]
		n = np.shape(x)[0]
		self.model = linear_model.LinearRegression(fit_intercept=fit_intercept)
		x_xt = np.matmul(x, np.transpose(x)) / n
		eigen_values, eigen_vectors = largest_eigsh(x_xt, 10, which='LM')
		ev_diff = np.log(eigen_values[0:9]) - np.log(eigen_values[1:])
		k = int(np.minimum(np.argmin(ev_diff) + 1, 6))
		print(f'number of estimated factors: {10 - k}')
		est_factor = eigen_vectors[:, k:] * np.sqrt(n)
		self.loading = np.transpose(np.matmul(np.transpose(est_factor), x)) / n
		est_idiosyncratic = x - np.matmul(est_factor, np.transpose(self.loading))
		self.model.fit(est_factor, y)

	def predict(self, x):
		if self.model is not None:
			est_factor, _ = estimate_factor_structure_from_observation(x, self.loading)
			return self.model.predict(est_factor)
		else:
			raise ValueError("Please first fit the model")

	def fit_and_predict(self, x, y, valid_x, valid_y, test_x):
		self.fit(x, y, valid_x, valid_y, fit_intercept=False)
		return self.predict(test_x)


class FARM:
	def __init__(self, use_sp=True):
		self.model_pc = None
		self.model_sp = None
		self.use_sp = use_sp
		self.loading = None

	def fit(self, x, y, test_x, test_y, fit_intercept=False):
		# x: [n, p]
		# y: [n,]
		n = np.shape(x)[0]
		self.model_pc = linear_model.LinearRegression(fit_intercept=fit_intercept)
		x_xt = np.matmul(x, np.transpose(x)) / n
		eigen_values, eigen_vectors = largest_eigsh(x_xt, 10, which='LM')
		ev_diff = np.log(eigen_values[0:9]) - np.log(eigen_values[1:])
		k = int(np.minimum(np.argmin(ev_diff) + 1, 6))
		print(f'number of estimated factors: {10 - k}')
		est_factor = eigen_vectors[:, k:] * np.sqrt(n)
		self.loading = np.transpose(np.matmul(np.transpose(est_factor), x)) / n
		est_idiosyncratic = x - np.matmul(est_factor, np.transpose(self.loading))
		self.model_pc.fit(est_factor, y)
		res_y = y - self.model_pc.predict(est_factor)
		# print(f'[train] std y = {np.std(y)}, std res_y = {np.std(res_y)}')

		if self.use_sp:
			test_factor, test_idiosyncratic = estimate_factor_structure_from_observation(test_x, self.loading)
			test_res_y = test_y - self.model_pc.predict(test_factor)
			# print(f'[valid] std y = {np.std(test_y)}, std res_y = {np.std(test_res_y)}')
			self.model_sp = Lasso()
			self.model_sp.fit(est_idiosyncratic, res_y, test_idiosyncratic, test_res_y, fit_intercept=fit_intercept)

	def predict(self, x):
		if self.model_pc is not None:
			est_factor, est_idiosyncratic = estimate_factor_structure_from_observation(x, self.loading)
			y = self.model_pc.predict(est_factor)
			if self.model_sp is not None:
				y = y + self.model_sp.predict(est_idiosyncratic)
			return y
		else:
			raise ValueError("Please first fit the model")

	def fit_and_predict(self, x, y, valid_x, valid_y, test_x):
		self.fit(x, y, valid_x, valid_y, fit_intercept=False)
		return self.predict(test_x)