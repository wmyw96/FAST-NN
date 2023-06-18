from models.fast_nn import *
from models.far_nn import *
from data.fast_data import *
import torch
import numpy as np
from colorama import init, Fore
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from models.fast_nn import FactorAugmentedSparseThroughputNN
from models.far_nn import RegressionNN
from torch.utils.data import DataLoader
from utils import *


def calculate_predefined_matrix(unlabelled_x, r_bar):
	p = np.shape(unlabelled_x)[1]
	cov_mat = np.matmul(np.transpose(unlabelled_x), unlabelled_x)
	eigen_values, eigen_vectors = largest_eigsh(cov_mat, r_bar, which='LM')
	dp_matrix = eigen_vectors / np.sqrt(p)
	estimate_f = np.matmul(unlabelled_x, dp_matrix)
	cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
	cov_fx_mat = np.matmul(np.transpose(estimate_f), unlabelled_x)
	rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)
	return dp_matrix, rs_matrix


def train_loop(data_loader, model, loss_fn, optimizer, reg_lambda, reg_tau):
	loss_rec = {'l2_loss': 0.0}
	if reg_tau is not None:
		loss_rec['reg_loss'] = 0.0
	for batch, (x, y) in enumerate(data_loader):
		pred = model(x, is_training=True)
		loss = loss_fn(pred, y)
		loss_rec['l2_loss'] += loss.item()
		if reg_tau is not None:
			reg_loss = model.regularization_loss(reg_tau, True)
			loss_rec['reg_loss'] += reg_lambda * reg_loss
			loss += reg_lambda * reg_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	loss_rec['l2_loss'] /= len(data_loader)
	if reg_tau is not None:
		loss_rec['reg_loss'] /= len(data_loader)
	return loss_rec


def test_loop(data_loader, model, loss_fn, reg_lambda, reg_tau):
	loss_sum = 0
	with torch.no_grad():
		for x, y in data_loader:
			pred = model(x, is_training=False)
			loss_sum += loss_fn(pred, y).item()
	loss_rec = {'l2_loss': loss_sum / len(data_loader)}
	if reg_tau is not None:
		loss_rec['reg_loss'] = reg_lambda * model.regularization_loss(reg_tau, True)
	return loss_rec


class NNEstimator:
	def __init__(self, r_bar=4):
		self.model = True
		self.r_bar = r_bar
		self.learning_rate = 1e-3
		self.num_epoch = 300
		self.model_color = Fore.YELLOW
		self.depth = 3
		self.width = 32
		self.hp_tau = 1e-1
		self.n_ensemble = 1
		self.p = None
		self.choice_lambda = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
							  0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
		#self.choice_lambda = [0]
		self.dp_matrix = None
		self.rs_matrix = None
		self.choice_large_lambda = [0.2, 0.5, 1]
		self.choice_small_lambda = [0.001, 0.0005, 0.0002, 0.0001]

	def single_fit_and_predict(self, train_data_loader, valid_data_loader, test_x, reg_lambda):
		device = "cuda" if torch.cuda.is_available() else "cpu"
		nn_model = \
			FactorAugmentedSparseThroughputNN(p=self.p, r_bar=self.r_bar, depth=self.depth,
											width=self.width, sparsity=self.r_bar,
											dp_mat=self.dp_matrix, rs_mat=self.rs_matrix).to(device)
		anneal_rate = (self.hp_tau * 10 - self.hp_tau) / self.num_epoch
		anneal_tau = self.hp_tau * 10

		mse_loss = nn.MSELoss()
		optimizer = torch.optim.Adam(nn_model.parameters(), lr=self.learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

		cur_valid = 1e9
		last_update = 1e9
		for epoch in range(self.num_epoch):
			anneal_tau -= anneal_rate
			train_losses = train_loop(train_data_loader, nn_model, mse_loss, optimizer, reg_lambda, anneal_tau)
			scheduler.step()
			valid_losses = test_loop(valid_data_loader, nn_model, mse_loss, reg_lambda, anneal_tau)
			if valid_losses['l2_loss'] < cur_valid:
				cur_valid = valid_losses['l2_loss']
				last_update = epoch
				with torch.no_grad():
					test_y = nn_model(torch.tensor(test_x, dtype=torch.float32)).detach().numpy()
		print(f'[FAST-NN] lambda = {reg_lambda}, last_update = {last_update}, valid loss = {cur_valid}')
		return cur_valid, test_y

	def model_fit_and_predict(self, x, y, valid_x, valid_y, test_x, candidate_lambda):
		dp_matrix, rs_matrix = calculate_predefined_matrix(x, self.r_bar)
		# x shape = [n, p]
		# y shape = [p]
		self.p = np.shape(x)[1]
		self.dp_matrix, self.rs_matrix = dp_matrix, rs_matrix
		y_ex = np.reshape(y, (np.shape(y)[0], 1))
		valid_y_ex = np.reshape(valid_y, (np.shape(valid_y)[0], 1))

		# build dataset
		torch_train = RegressionDataset(x, y_ex)
		train_data_loader = DataLoader(torch_train, batch_size=np.shape(x)[0] // 4)
		torch_valid = RegressionDataset(valid_x, valid_y_ex)
		valid_data_loader = DataLoader(torch_valid, batch_size=np.shape(valid_x)[0])

		best_valid = 1e9
		best_lambda = None
		test_y = None
		for reg_lambda in candidate_lambda:
			# create model
			valid_error = 0.0
			y_pred = 0.0
			for t in range(self.n_ensemble):
				valid_error_, y_pred_ = \
					self.single_fit_and_predict(train_data_loader, valid_data_loader, test_x, reg_lambda)
				valid_error += valid_error_
				y_pred += y_pred_
			valid_error /= self.n_ensemble
			y_pred /= self.n_ensemble
			if best_valid > valid_error:
				best_valid = valid_error
				best_lambda = reg_lambda
				test_y = y_pred
		return best_valid, best_lambda, test_y

	def fit_and_predict(self, x, y, valid_x, valid_y, test_x):
		best_valid, best_lambda, test_y = self.model_fit_and_predict(x, y, valid_x, valid_y, test_x, self.choice_lambda)
		test_y = np.reshape(test_y, (np.shape(test_y)[0],))
		print(f"(FAST-NN Estimator) best alpha = {best_lambda}, valid mse = {best_valid}")
		return test_y
