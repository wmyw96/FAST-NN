import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# plt.style.use('fivethirtyeight')

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=True)

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#9acdc4',  # pain blue
	'#05348b',  # dark blue
	'#6bb392',
	'#e5a84b',
]
cand_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
l2_loss_matrix_mn = np.zeros((len(cand_p), 6))
l2_loss_matrix_std = np.zeros((len(cand_p), 6))

for i, p in enumerate(cand_p):
	results = []
	for s in range(0, 200):
		try:
			results.append(genfromtxt(f"../logs/exp1-old/p{p}s{s}.csv", delimiter=','))
		except:
			print(f"Load Data Error: no record p = {p}, s = {s}")
	result = np.array(results)
	l2_loss_matrix_mn[i, :] = np.mean(result, axis=0)
	#l2_loss_matrix_std[i, :] = np.std(result, axis=0)


model_name = [
	'Oracle-NN',
	'FAR-NN',
	'Vanilla-NN',
	'NN-Joint',
	'PCA-A-NN',
	'FAST-NN',
]

lines = [
	'dashed',
	'solid',
	'solid',
	'solid',
	'solid',
	'solid'
]

marker = [
	',',
	'8',
	'v',
	'^',
	's',
	'x',
]

plt.figure(figsize=(6, 6))
for i in [0, 1, 5, 4, 2, 3]:
	plt.plot(cand_p, l2_loss_matrix_mn[:, i], color=color_tuple[i], linestyle=lines[i], label=model_name[i],
			marker=marker[i])

	plt.fill_between(np.array(cand_p), l2_loss_matrix_mn[:, i] - l2_loss_matrix_std[:, i],
 					l2_loss_matrix_mn[:, i] + l2_loss_matrix_std[:, i], color=color_tuple[i], alpha=0.1)

plt.ylabel(r"$\widehat{\mathtt{MSE}}$")
plt.xlabel(r"ambient dimension $p$")

plt.yscale("log")
plt.xscale("log")
plt.ylim([0.06, 0.75])
plt.yticks([0.08, 0.1, 0.2, 0.3, 0.6], ['0.08', '0.1', '0.2', '0.3', '0.6'])
plt.legend()
plt.show()
