from colorama import init, Fore
import torch
import random
import numpy as np
from torch import nn
from utils import unpack_loss
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from data.covariate import FactorModel
from models.fanam import SparseNeuralAdditiveModels
from models.far_nn import RegressionNN
from data.fast_data import AdditiveModel, RegressionDataset
from torch.utils.data import DataLoader

import argparse
import time

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples", type=int, default=300)
parser.add_argument("--p", help="data dimension", type=int, default=200)
parser.add_argument("--s", help="number of important variables", type=int, default=5)
parser.add_argument("--hp_lambda", help="hyperparameter lambda", type=float, default=1)
parser.add_argument("--width", help="width of NN", type=int, default=64)
parser.add_argument("--depth", help="depth of NN", type=int, default=3)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=2)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
args = parser.parse_args()

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
n_test = 500
n_valid = args.n * 3 // 10

# data generating process
regression_model = AdditiveModel(num_funcs=args.s, normalize=False)
print(regression_model)
xm = FactorModel(p=args.p, r=0, b_f=1, b_u=1)


def additive_data(n: int, noise_level=0.0):
	observation = xm.sample(n=n, latent=False)
	# print(np.shape(observation[:, :args.s]))
	x, y = observation, regression_model.sample(observation[:, :args.s])
	noise = np.random.normal(0, noise_level, (n, 1))
	return x, y + noise


x_train_obs, y_train = additive_data(n_train, 0.3)
x_valid_obs, y_valid = additive_data(n_valid, 0.3)
x_test_obs, y_test = additive_data(n_test, 0)

train_obs_data = RegressionDataset(x_train_obs, y_train)
train_obs_dataloader = DataLoader(train_obs_data, batch_size=batch_size, shuffle=True)

test_obs_data = RegressionDataset(x_test_obs, y_test)
test_obs_dataloader = DataLoader(test_obs_data, batch_size=batch_size, shuffle=True)

valid_obs_data = RegressionDataset(x_valid_obs, y_valid)
valid_obs_dataloader = DataLoader(valid_obs_data, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
sparse_nam_model = \
	SparseNeuralAdditiveModels(p=args.p, depth=depth, width=width).to(device)
vanilla_nn_model = \
	RegressionNN(d=args.p, depth=depth, width=width*args.p).to(device)


learning_rate = 5 * 1e-4
num_epoch = 300


def train_loop(data_loader, model, loss_fn, optimizer, l1_reg=False, anneal_rate=1.0):
	loss_rec = {'l2_loss': 0.0}
	if l1_reg:
		loss_rec['reg_loss'] = 0.0
		loss_rec['overall_loss'] = 0.0
	for batch, (x, y) in enumerate(data_loader):
		if l1_reg:
			pred = model(x, is_training=True, anneal=anneal_rate)
		else:
			pred = model(x, is_training=True)
		loss = loss_fn(pred, y)
		loss_rec['l2_loss'] += loss.item()
		if l1_reg:
			loss_rec['reg_loss'] += model.regularization_loss().item() * args.hp_lambda
			loss = loss + args.hp_lambda * model.regularization_loss()
			loss_rec['overall_loss'] += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	loss_rec['l2_loss'] /= len(data_loader)
	if l1_reg:
		loss_rec['reg_loss'] /= len(data_loader)
		loss_rec['overall_loss'] /= len(data_loader)
	return loss_rec


def test_loop(data_loader, model, loss_fn, l1_reg=False, anneal_rate=1.0):
	loss_sum = 0
	with torch.no_grad():
		for x, y in data_loader:
			if l1_reg:
				pred = model(x, is_training=False, anneal=anneal_rate)
			else:
				pred = model(x, is_training=False)
			loss_sum += loss_fn(pred, y).item()
	loss_rec = {'l2_loss': loss_sum / len(data_loader)}
	return loss_rec


mse_loss = nn.MSELoss()
models = {
	'sparse-nam': sparse_nam_model,
	'oracle-nn': vanilla_nn_model
}

optimizers = {}
for method_name, model_x in models.items():
	optimizer_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate)
	optimizers[method_name] = optimizer_x


if True:
	model_names = ['sparse-nam', 'oracle-nn']
	colors = [Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.LIGHTRED_EX, Fore.LIGHTYELLOW_EX,
			  Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTCYAN_EX]
	best_valid, model_color = {}, {}
	for i, name in enumerate(model_names):
		best_valid[name] = 1e9
		model_color[name] = colors[i]
	test_perf = {}
	anneal_rate = 1.0
	for epoch in range(num_epoch):
		anneal_rate *= (1/0.99)
		if epoch % 10 == 0:
			print(f"Epoch {epoch}\n--------------------")
		for model_name in model_names:
			use_reg = model_name == 'sparse-nam'
			train_losses = \
				train_loop(train_obs_dataloader, models[model_name], mse_loss,
						   optimizers[model_name], use_reg, anneal_rate)
			valid_losses = test_loop(valid_obs_dataloader, models[model_name], mse_loss, use_reg, anneal_rate)
			if valid_losses['l2_loss'] < best_valid[model_name]:
				best_valid[model_name] = valid_losses['l2_loss']
				test_losses = test_loop(test_obs_dataloader, models[model_name], mse_loss, use_reg, anneal_rate)
				test_perf[model_name] = test_losses['l2_loss']
				print(model_color[model_name] + f"Model [{model_name}]: update test loss, "
												f"best valid loss = {valid_losses['l2_loss']}, "
												f"current test loss = {test_losses['l2_loss']}")
				if use_reg:
					with torch.no_grad():
						beta_weight = torch.nn.functional.softmax(models[model_name].beta_logits * anneal_rate, dim=1)
						print(beta_weight.detach().numpy())
			if epoch % 10 == 0:
				print(f"Model [{model_name}]: \n    (Train)  " + unpack_loss(train_losses) +
					"\n    (Valid)  " + unpack_loss(valid_losses))

	result = np.zeros((1, len(model_names)))
	for i, name in enumerate(model_names):
		result[0, i] = test_perf[name]
	# return result
