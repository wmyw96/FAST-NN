from colorama import init, Fore
import torch
import random
import numpy as np
from torch import nn

from data.covariate import FactorModel
from models.far_nn import FactorAugmentedNN, RegressionNN
from models.fast_nn import FactorAugmentedSparseThroughputNN
from data.fast_data import AdditiveModel, RegressionDataset
from torch.utils.data import DataLoader
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

import argparse
import time

# TODO Re-run exp1

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples", type=int, default=500)
parser.add_argument("--m", help="number of samples to calculate the diversified "
                                "projection matrix", type=int, default=50)
parser.add_argument("--p", help="data dimension", type=int, default=1000)
parser.add_argument("--r", help="factor dimension", type=int, default=5)
parser.add_argument("--r_bar", help="diversified weight dimension", type=int, default=10)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=4)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=1)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
parser.add_argument("--exp_id", help="exp id", type=int, default=1)
parser.add_argument("--record_dir", help="directory to save record", type=str, default="")
parser.add_argument("--hp_lambda", help="hyperparameter lambda", type=float, default=1.3)
parser.add_argument("--hp_tau", help="hyperparameter tau", type=float, default=0.005)

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

penalty_weight = np.log(p) / n_train * args.hp_lambda
# data generating process
regression_model = AdditiveModel(num_funcs=args.r, normalize=False)
lfm = FactorModel(p=p, r=args.r, b_f=1, b_u=1)
print(regression_model)


observation, factor, _ = lfm.sample(n=10000, latent=True)
x, y = observation, regression_model.sample(factor)
v1 = np.std(y)
print(v1)


def far_data(n, noise_level=0.0):
    observation, factor, _ = lfm.sample(n=n, latent=True)
    x, y = observation, regression_model.sample(factor)
    noise = np.random.normal(0, noise_level, (n, 1))
    return x, factor, y + noise


# prepare dataset
x_train_obs, x_train_latent, y_train = far_data(n_train, 0.3)
x_valid_obs, x_valid_latent, y_valid = far_data(n_valid, 0.3)
x_test_obs, x_test_latent, y_test = far_data(n_test, 0)

train_obs_data = RegressionDataset(x_train_obs, y_train)
train_latent_data = RegressionDataset(x_train_latent, y_train)
train_obs_dataloader = DataLoader(train_obs_data, batch_size=batch_size, shuffle=True)
train_latent_dataloader = DataLoader(train_latent_data, batch_size=batch_size, shuffle=True)

test_obs_data = RegressionDataset(x_test_obs, y_test)
test_latent_data = RegressionDataset(x_test_latent, y_test)
test_obs_dataloader = DataLoader(test_obs_data, batch_size=batch_size, shuffle=True)
test_latent_dataloader = DataLoader(test_latent_data, batch_size=batch_size, shuffle=True)

valid_obs_data = RegressionDataset(x_valid_obs, y_valid)
valid_latent_data = RegressionDataset(x_valid_latent, y_valid)
valid_obs_dataloader = DataLoader(valid_obs_data, batch_size=batch_size, shuffle=True)
valid_latent_dataloader = DataLoader(valid_latent_data, batch_size=batch_size, shuffle=True)

# model

unlabelled_x, _, _ = far_data(args.m)
cov_mat = np.matmul(np.transpose(unlabelled_x), unlabelled_x)
eigen_values, eigen_vectors = largest_eigsh(cov_mat, args.r_bar, which='LM')
dp_matrix = eigen_vectors / np.sqrt(p)
eigen_values_oracler, eigen_vectors_oracler = largest_eigsh(cov_mat, args.r, which='LM')
dp_matrix_oracler = eigen_vectors_oracler / np.sqrt(p)
estimate_f = np.matmul(unlabelled_x, dp_matrix)
cov_f_mat = np.matmul(np.transpose(estimate_f), estimate_f)
cov_fx_mat = np.matmul(np.transpose(estimate_f), unlabelled_x)
rs_matrix = np.matmul(np.linalg.pinv(cov_f_mat), cov_fx_mat)

print(f"Diversified projection matrix size {np.shape(dp_matrix)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
far_nn_model = FactorAugmentedNN(p=args.p, r_bar=args.r_bar, depth=depth, width=width,
                                 dp_mat=dp_matrix, fix_dp_mat=True).to(device)
fast_nn_model = \
    FactorAugmentedSparseThroughputNN(p=args.p, r_bar=args.r_bar, depth=depth, width=width,
                                        sparsity=args.r_bar, dp_mat=dp_matrix, rs_mat=rs_matrix).to(device)
pca_nn_model = \
    FactorAugmentedNN(p=args.p, r_bar=args.r, depth=depth, width=width, dp_mat=dp_matrix_oracler,
                      fix_dp_mat=True, with_x=True).to(device)

oracle_nn_model = RegressionNN(d=args.r, depth=depth, width=width).to(device)
vanilla_nn_model = RegressionNN(d=args.p, depth=depth, width=width).to(device)
far_joint_nn_model = FactorAugmentedNN(p=args.p, r_bar=args.r_bar, depth=depth, width=width,
                                       dp_mat=dp_matrix, fix_dp_mat=False).to(device)
far_joint_dropout_nn_model = FactorAugmentedNN(p=args.p, r_bar=args.r_bar, depth=depth, width=width,
                                               dp_mat=dp_matrix, fix_dp_mat=False, input_dropout=True,
                                               dropout_rate=args.dropout_rate).to(device)
dropout_nn_model = RegressionNN(d=args.p, depth=depth, width=width, input_dropout=True,
                                dropout_rate=args.dropout_rate).to(device)

print(f"FAR-NN Model:\n {far_nn_model}")
print(f"Oracle-NN Model:\n {oracle_nn_model}")
print(f"Vanilla-NN Model:\n {vanilla_nn_model}")
print(f"FAR-NN Joint Training Model:\n {far_joint_nn_model}")

# training configurations
learning_rate = args.lr
num_epoch = 200


def train_loop(data_loader, model, loss_fn, optimizer, reg_tau=None):
    loss_sum = 0
    for batch, (x, y) in enumerate(data_loader):
        pred = model(x, is_training=True)
        loss = loss_fn(pred, y)
        loss_sum += loss.item()
        if reg_tau is not None:
            reg_loss = model.regularization_loss(reg_tau)
            loss += penalty_weight * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_sum / len(data_loader)


def test_loop(data_loader, model, loss_fn):
    loss_sum = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred = model(x, is_training=False)
            loss_sum += loss_fn(pred, y).item()
    return loss_sum / len(data_loader)


mse_loss = nn.MSELoss()
models = {
    'far-nn': far_nn_model,
    'fast-nn': fast_nn_model,
    'pca-aug-nn': pca_nn_model,
    'oracle-nn': oracle_nn_model,
    'vanilla-nn': vanilla_nn_model,
    'joint-nn': far_joint_nn_model,
    'joint-dropout-nn': far_joint_dropout_nn_model,
    'dropout-nn': dropout_nn_model
}

optimizers = {}
for method_name, model_x in models.items():
    optimizer_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate)
    optimizers[method_name] = optimizer_x


def train_one_dim_nn():
    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}\n-------------------")
        train_data_loader = train_latent_dataloader
        train_loss = train_loop(train_data_loader, models["oracle-nn"], mse_loss, optimizers["oracle-nn"])
        test_data_loader = test_latent_dataloader
        test_loss = test_loop(test_data_loader, models["oracle-nn"], mse_loss)
        print(f" Model [oracle-nn]: train L2 error = {train_loss}, test L2 error = {test_loss}")
        # visualize output
        with torch.no_grad():
            x, y = next(iter(test_data_loader))
            pred = models['oracle-nn'](x)
            if epoch % 5 == 0:
                plt.figure(figsize=(16, 8))
                plt.scatter(x.numpy(), y.numpy(), color='red')
                plt.scatter(x.numpy(), pred.numpy(), color=palette(2))
                plt.savefig(f"figures/epoch{epoch}.pdf")
                plt.close()


def joint_train(model_names):
    colors = [Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.LIGHTRED_EX, Fore.LIGHTYELLOW_EX,
              Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTCYAN_EX]
    best_valid, model_color = {}, {}
    for i, name in enumerate(model_names):
        best_valid[name] = 1e9
        model_color[name] = colors[i]
    test_perf = {}
    anneal_rate = (args.hp_tau * 20 - args.hp_tau) / num_epoch
    anneal_tau = args.hp_tau * 20
    train_result, valid_result = np.zeros((num_epoch, len(model_names))), np.zeros((num_epoch, len(model_names)))
    early_stopping = [0] * len(model_names)
    for epoch in range(num_epoch):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}\n--------------------")
        anneal_tau -= anneal_rate
        for i, model_name in enumerate(model_names):
            reg_tau = anneal_tau if (model_name == 'fast-nn') else None
            train_data_loader = train_latent_dataloader if (model_name == 'oracle-nn') else train_obs_dataloader
            train_loss = train_loop(train_data_loader, models[model_name], mse_loss, optimizers[model_name], reg_tau)
            valid_data_loader = valid_latent_dataloader if (model_name == 'oracle-nn') else valid_obs_dataloader
            valid_loss = test_loop(valid_data_loader, models[model_name], mse_loss)
            train_result[epoch, i], valid_result[epoch, i] = train_loss, valid_loss
            if valid_loss < best_valid[model_name] and epoch >= 100:
                early_stopping[i] = epoch
                best_valid[model_name] = valid_loss
                test_data_loader = test_latent_dataloader if (model_name == 'oracle-nn') else test_obs_dataloader
                test_loss = test_loop(test_data_loader, models[model_name], mse_loss)
                test_perf[model_name] = test_loss
                print(model_color[model_name] + f"Model [{model_name}]: update test loss, "
                                                f"best valid loss = {valid_loss}, current test loss = {test_loss}")
            if epoch % 10 == 0:
                print(f"Model [{model_name}]: train L2 error = {train_loss}, valid L2 error = {valid_loss}")
    result = np.zeros((1, len(model_names)))
    for i, name in enumerate(model_names):
        result[0, i] = test_perf[name]
    return result, train_result, valid_result, early_stopping

timestep = np.arange(num_epoch)
print(timestep)
_, trainc, validc, early_stopping = joint_train(["oracle-nn", "far-nn", "fast-nn", "pca-aug-nn", "joint-nn", "vanilla-nn"])

color_tuple = [
    '#ae1908',  # red
    '#ec813b',  # orange
    '#e5a84b',
    '#6bb392',
    '#05348b',  # dark blue
    '#9acdc4',  # pain blue
]
model_name = [
    'Oracle-NN',
    'FAR-NN',
    'FAST-NN',
    'PCA-A-NN',
    'NN-Joint',
    'Vanilla-NN',
]

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=True)

plt.figure(figsize=(6, 6))
for i in range(6):
    plt.plot(timestep, trainc[:, i], color=color_tuple[i], linestyle='dashed', linewidth=1.8)
    plt.plot(timestep, validc[:, i], color=color_tuple[i], linestyle='solid', label=model_name[i], linewidth=1.8)

#for i in range(6):
#    plt.vlines(x=early_stopping[i], ymin=-0.1, ymax=2.5, color=color_tuple[i], linestyle='dotted', linewidth=1)

plt.ylabel(r"Empirical $L_2$ Loss")
plt.xlabel(r"Epochs")
plt.ylim([-0.1, 2.3])
#plt.yscale("log")
plt.legend()
plt.show()

