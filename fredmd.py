from data.fredmd_data import fred_md_data
from utils import *
from stat_methods import *
from fast_nn_estimator import *
import numpy as np
import random
import time
from colorama import init, Fore


corpus = fred_md_data('data/FRED-MD/transformed_modern.csv')
n = corpus.valid_n
print(n)
seed = 4869
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def get_index_array(l, r):
	index = []
	for i in range(r - l + 1):
		index.append(i + l)
	return index


def split_x_y(data_corpus, idx):
	x = np.concatenate([data_corpus[:, :idx], data_corpus[:, idx + 1:]], 1)
	y = data_corpus[:, idx]
	return x, y


# cross-window test
window_size = 200

p = np.shape(corpus.data)[1]
n_windows = n - window_size
y_value = np.zeros((p, n_windows, 6))
r2 = np.zeros((p, 4))

model_names = ['FAST', 'FARM', 'Lasso', 'PCR']

start_time = time.time()
init(autoreset=True)

for pred_idx in range(p):
	corpus = fred_md_data('data/FRED-MD/transformed_modern.csv', pred_index=pred_idx)
	print(f'======================================== {pred_idx} ========================================')
	rss = {}
	for model_name in model_names:
		rss[model_name] = 0.0
	tss = 0.0
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)

	if True:
		i = 0
		idx = 0
		train, valid, test, mn_stat, std_stat = \
			corpus.get_split_data(get_index_array(i, i + window_size - 1),
								  get_index_array(i + window_size, corpus.valid_n - 1))
		y_std = std_stat[0, idx]
		y_mn = mn_stat[0, idx]
		train_x, train_y = split_x_y(train, idx)
		assert np.abs(np.mean(train_y)) < 1e-9
		valid_x, valid_y = split_x_y(valid, idx)
		test_x, test_y = split_x_y(test, idx)

		# exact value of y
		tss += np.mean(np.square(test_y - y_mn)) * (y_std ** 2)

		info_str = "R^2 stat:  "
		#y_value[pred_idx, i, 0], y_value[pred_idx, i, 1] = test_y * y_std + y_mn, y_mn

		for k, model_name in enumerate(model_names):
			model = None
			if model_name == 'FARM':
				model = FARM()
			if model_name == 'Lasso':
				model = Lasso()
			if model_name == 'PCR':
				model = PCR()
			if model_name == 'FAST':
				model = NNEstimator(4)

			pred = model.fit_and_predict(train_x, train_y, valid_x, valid_y, test_x) # * y_std + y_mn
			rss[model_name] += np.mean(np.square(test_y - pred)) * (y_std ** 2)
			print(f'timestep {i}, true = {test_y}, model ({model_name}) = {pred}')
			info_str += f"({model_name}) {1 - rss[model_name]/tss}    "
			#y_value[pred_idx, i, k + 2] = pred * y_std + y_mn
			# for select important variable p
			# r2[pred_idx, k] = 1 - rss[model_name]/tss
			r2[pred_idx, k] = 1 - rss[model_name] / tss
		print(Fore.YELLOW + info_str)
	np.savetxt('r2_pred.txt', r2)

print(f'End: time = {time.time() - start_time}')
#np.save('y_value.npy', y_value)
