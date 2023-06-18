from colorama import init, Fore
import torch
import random
import numpy as np
from torch import nn
from utils import unpack_loss, visualize_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data.covariate import FactorModel
from models.fast_nn import FactorAugmentedSparseThroughputNN
from models.far_nn import FactorAugmentedNN, RegressionNN
from data.fast_data import RegressionDataset, AdditiveModel, HierarchicalCompositionModels
from torch.utils.data import DataLoader
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

import argparse
import time

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--m", help="number of samples to calculate the diversified "
								"projection matrix", type=int, default=200)
parser.add_argument("--p", help="data dimension", type=int, default=1000)
parser.add_argument("--r", help="factor dimension", type=int, default=4)
parser.add_argument('--s', help="number of important individual variables", type=int, default=5)
parser.add_argument("--hp_lambda", help="hyperparameter lambda", type=float, default=0.0108)
parser.add_argument("--hp_tau", help="hyperparameter tau", type=float, default=0.005)
parser.add_argument("--r_bar", help="diversified weight dimension", type=int, default=10)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=4)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=2)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--hcm_id", help="hcm id", type=int, default=0)
parser.add_argument("--record_dir", help="directory to save record", type=str, default="")
parser.add_argument("--visualize_mat", help="whether to visualize the variable selection matrix",
					type=bool, default=False)
args = parser.parse_args()

start_time = time.time()

# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# hyper-parameters
n_train = args.n
p = args.p
batch_size = args.batch_size
width = args.width
depth = args.depth
n_test = 10000
n_valid = args.n * 3 // 10
penalty_weight = np.log(p) / n_train * 1.3 #* args.hp_lambda
#print('n = {}, log(p) = {}, penalty weight = {}'.format(n_train, np.log(p), penalty_weight))
# data generating process
#regression_model = AdditiveModel(num_funcs=args.r + args.s, normalize=False)
#regression_model.func_idx[4] = 0
#regression_model.func_idx[5] = 8

'''
regression_model.func_idx[5] = 5
regression_model.func_idx[6] = 0
regression_model.func_idx[7] = 5
regression_model.func_idx[8] = 0
'''
regression_model = HierarchicalCompositionModels(args.hcm_id)

lfm = FactorModel(p=p, r=args.r, b_f=1, b_u=1)
print(regression_model)


def fast_data(n, s, noise_level=0.0):
	observation, factor, idiosyncratic_error = lfm.sample(n=n, latent=True)
	related_variables = np.concatenate([factor, idiosyncratic_error[:, :s]], 1)
	x, y = observation, regression_model.sample(related_variables)
	print(f"fast data x {np.shape(x)}, y {np.shape(y)}")
	noise = np.random.normal(0, noise_level, (n, 1))
	return x, factor, related_variables, y + noise


# prepare dataset
x_train_obs, x_train_factor, x_train_latent, y_train = fast_data(n_train, args.s, 0.3)
x_valid_obs, x_valid_factor, x_valid_latent, y_valid = fast_data(n_valid, args.s, 0.3)
x_test_obs, x_test_factor, x_test_latent, y_test = fast_data(n_test, args.s, 0)

train_obs_data = RegressionDataset(x_train_obs, y_train)
train_factor_data = RegressionDataset(x_train_factor, y_train)
train_latent_data = RegressionDataset(x_train_latent, y_train)
train_obs_dataloader = DataLoader(train_obs_data, batch_size=batch_size, shuffle=True)
train_factor_dataloader = DataLoader(train_factor_data, batch_size=batch_size, shuffle=True)
train_latent_dataloader = DataLoader(train_latent_data, batch_size=batch_size, shuffle=True)

test_obs_data = RegressionDataset(x_test_obs, y_test)
test_factor_data = RegressionDataset(x_test_factor, y_test)
test_latent_data = RegressionDataset(x_test_latent, y_test)
test_obs_dataloader = DataLoader(test_obs_data, batch_size=batch_size, shuffle=True)
test_factor_dataloader = DataLoader(test_factor_data, batch_size=batch_size, shuffle=True)
test_latent_dataloader = DataLoader(test_latent_data, batch_size=batch_size, shuffle=True)

valid_obs_data = RegressionDataset(x_valid_obs, y_valid)
valid_factor_data = RegressionDataset(x_valid_factor, y_valid)
valid_latent_data = RegressionDataset(x_valid_latent, y_valid)

valid_obs_dataloader = DataLoader(valid_obs_data, batch_size=batch_size, shuffle=True)
valid_factor_dataloader = DataLoader(valid_factor_data, batch_size=batch_size, shuffle=True)
valid_latent_dataloader = DataLoader(valid_latent_data, batch_size=batch_size, shuffle=True)

# model

unlabelled_x, oracle_f, oracle_u = lfm.sample(n=args.m, latent=True)
cov_mat = np.matmul(np.transpose(unlabelled_x), unlabelled_x)
eigen_values, eigen_vectors = largest_eigsh(cov_mat, args.r_bar, which='LM')
dp_matrix = eigen_vectors / np.sqrt(p)
estimate_f = np.matmul(unlabelled_x, dp_matrix)
cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
cov_fx_mat = np.matmul(np.transpose(estimate_f), unlabelled_x)
rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)

# test u estimation accuracy
utest_x, utest_f, utest_u = lfm.sample(n=1000, latent=True)
estimate_u = utest_x - np.matmul(np.matmul(utest_x, dp_matrix), rs_matrix)
error = np.max(np.abs(utest_u - estimate_u))

print(f"unit test: testing the error of estimating u: {error}")
# raise ValueError("Temp")

print(f"Diversified projection matrix size {np.shape(dp_matrix)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
far_nn_model = FactorAugmentedNN(p=args.p, r_bar=args.r_bar, depth=depth, width=width,
									dp_mat=dp_matrix, fix_dp_mat=True).to(device)

fast_nn_model = \
	FactorAugmentedSparseThroughputNN(p=args.p, r_bar=args.r_bar, depth=depth, width=width,
									  sparsity=args.r_bar, dp_mat=dp_matrix, rs_mat=rs_matrix).to(device)
oracle_nn_model = \
	RegressionNN(d=args.r + args.s, depth=depth, width=width).to(device)
oracle_f_nn_model = \
	RegressionNN(d=args.r, depth=depth, width=width).to(device)

print(f"FAST-NN Model:\n {fast_nn_model}")
print(f"Oracle-NN Model:\n {oracle_nn_model}")
print(f"Oracle-F-NN Model:\n {oracle_f_nn_model}")

learning_rate = 1e-3
num_epoch = 100


def train_loop(data_loader, model, loss_fn, optimizer, reg_tau=None):
	loss_rec = {'l2_loss': 0.0}
	if reg_tau is not None:
		loss_rec['reg_loss'] = 0.0
	for batch, (x, y) in enumerate(data_loader):
		pred = model(x, is_training=True)
		loss = loss_fn(pred, y)
		loss_rec['l2_loss'] += loss.item()
		if reg_tau is not None:
			reg_loss = model.regularization_loss(reg_tau)
			loss_rec['reg_loss'] += penalty_weight * reg_loss
			loss += penalty_weight * reg_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	loss_rec['l2_loss'] /= len(data_loader)
	if reg_tau is not None:
		loss_rec['reg_loss'] /= len(data_loader)
	return loss_rec


def test_loop(data_loader, model, loss_fn, reg_tau=None):
	loss_sum = 0
	with torch.no_grad():
		for x, y in data_loader:
			pred = model(x, is_training=False)
			loss_sum += loss_fn(pred, y).item()
	loss_rec = {'l2_loss': loss_sum / len(data_loader)}
	if reg_tau is not None:
		loss_rec['reg_loss'] = args.hp_lambda * model.regularization_loss(reg_tau)
	return loss_rec


mse_loss = nn.MSELoss()
models = {
	'fast-nn': fast_nn_model,
	'far-nn': far_nn_model,
	'oracle-nn': oracle_nn_model,
	'oracle-f-nn': oracle_f_nn_model
}


optimizers = {}
schedulers = {}
for method_name, model_x in models.items():
	optimizer_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate)
	optimizers[method_name] = optimizer_x
	schedulers[method_name] = torch.optim.lr_scheduler.ExponentialLR(optimizers[method_name], gamma=0.995)


def choose_dataloader(model_name, data_loaders):
	if model_name == 'oracle-nn':
		return data_loaders[1]
	if model_name == 'oracle-f-nn':
		return data_loaders[2]
	return data_loaders[0]


def joint_train(model_names):
	# model_names = ['oracle-nn', 'fast-nn', 'oracle-f-nn']
	colors = [Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.LIGHTRED_EX, Fore.LIGHTYELLOW_EX,
			  Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTCYAN_EX]
	best_valid, model_color = {}, {}
	for i, name in enumerate(model_names):
		best_valid[name] = 1e9
		model_color[name] = colors[i]
	test_perf = {}
	anneal_rate = (args.hp_tau * 20 - args.hp_tau) / num_epoch
	anneal_tau = args.hp_tau * 20
	variable_selection_mat = None
	for epoch in range(num_epoch):
		if epoch % 10 == 0:
			print(f"Epoch {epoch}\n--------------------")
		anneal_tau -= anneal_rate
		for model_name in model_names:
			reg_tau = anneal_tau if (model_name == 'fast-nn') else None
			train_data_loader = \
				choose_dataloader(model_name, [train_obs_dataloader, train_latent_dataloader, train_factor_dataloader])
			train_losses = train_loop(train_data_loader, models[model_name], mse_loss, optimizers[model_name], reg_tau)
			schedulers[model_name].step()
			valid_data_loader = \
				choose_dataloader(model_name, [valid_obs_dataloader, valid_latent_dataloader, valid_factor_dataloader])
			valid_losses = test_loop(valid_data_loader, models[model_name], mse_loss, reg_tau)
			if valid_losses['l2_loss'] < best_valid[model_name]:
				best_valid[model_name] = valid_losses['l2_loss']
				test_data_loader = \
					choose_dataloader(model_name,
									  [test_obs_dataloader, test_latent_dataloader, test_factor_dataloader])
				test_losses = test_loop(test_data_loader, models[model_name], mse_loss)
				test_perf[model_name] = test_losses['l2_loss']
				print(model_color[model_name] + f"Model [{model_name}]: update test loss, "
												f"best valid loss = {valid_losses['l2_loss']}, "
												f"current test loss = {test_losses['l2_loss']}")
				if model_name == 'fast-nn':
					# visualize variable selection matrix
					variable_selection_mat = models[model_name].variable_selection.weight.detach().numpy()
					torch.save(models[model_name].state_dict(), f"temp-{model_name}")

			if epoch % 10 == 0:
				print(f"Model [{model_name}]: \n    (Train)  " + unpack_loss(train_losses) +
					"\n    (Valid)  " + unpack_loss(valid_losses))

	if args.visualize_mat:
		visualize_matrix(variable_selection_mat)
	# hcm3: python fast_exp.py --hcm_id 3 --p 1000 --visualize_mat True --seed 4 --m 200 --hp_tau 0.005
	# hcm1: standard

	result = np.zeros((1, len(model_names)))
	for i, name in enumerate(model_names):
		result[0, i] = test_perf[name]
	return result


test_l2_error = joint_train(['oracle-nn', 'fast-nn', 'far-nn', 'oracle-f-nn'])
if len(args.record_dir) > 0:
	np.savetxt(args.record_dir + f"/p{args.p}s{args.seed}.csv", test_l2_error, delimiter=",")
end_time = time.time()
print(f"Case with p = {args.p}, seed = {args.seed} done: time = {end_time - start_time} secs")