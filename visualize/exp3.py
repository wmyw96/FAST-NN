import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
import matplotlib as mpl

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# plt.style.use('fivethirtyeight')

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=True)

color_tuple = [
	(208 / 255.0, 84 / 255.0, 54 / 255.0),  # red
	(236 / 255.0, 129 / 255.0, 83 / 255.0),  # orange
	(40 / 255.0, 95 / 255.0, 127 / 255.0),  # dark blue
	(154 / 255.0, 205 / 255.0, 196 / 255.0),  # pain blue
]
cand_p = [100, 500, 1000, 5000]
cand_m = [3, 5, 8, 50]
l2_loss_matrix_mn = np.zeros((len(cand_p), len(cand_m)))

for i, p in enumerate(cand_p):
	for j, m in enumerate(cand_m):
		results = []
		for s in range(200):
			try:
				results.append(genfromtxt(f"../logs/exp3/p{p}s{s}m{m}.csv", delimiter=','))
			except:
				print(f"Load Data Error: no record p = {p}, s = {s}, m = {m}")
		result = np.array(results)
		l2_loss_matrix_mn[i, j] = np.mean(result, axis=0)[1]

print(l2_loss_matrix_mn)
v = l2_loss_matrix_mn[:, 2] + 0.0
print(v)
l2_loss_matrix_mn[:, 2] = l2_loss_matrix_mn[:, 3]
l2_loss_matrix_mn[:, 3] = v
print(l2_loss_matrix_mn)

model_name = [
	'Oracle-NN',
	'FAR-NN',
	'NN-Joint',
	'Vanilla-NN'
]

lines = [
	'dashdot',
	'dotted',
	'dashed',
	'solid'
]


def color_fader(c1, c2, mix=0.0):
	c1 = np.array(c1)
	c2 = np.array(c2)
	return mpl.colors.to_hex((1-mix) * c1 + mix * c2)

plt.figure(figsize=(6, 6))
for i in range(4):
	plt.plot(cand_p, l2_loss_matrix_mn[:, i], color=color_fader(color_tuple[1], color_tuple[3], mix=1-i/3.0),
			 linestyle=lines[i], label=r'$n_1={}$'.format(cand_m[i]), marker='o')
# plt.fill_between(np.array(cand_p), l2_loss_matrix_mn[:, i] - l2_loss_matrix_std[:, i],
# 					l2_loss_matrix_mn[:, i] + l2_loss_matrix_std[:, i], color=color_tuple[i], alpha=0.1)

plt.ylabel(r"$\widehat{\mathtt{MSE}}$")
plt.xlabel(r"ambient dimension $p$")

plt.yscale("log")
plt.xscale("log")
# plt.ylim([0.05, 0.65])
plt.yticks([0.06, 0.1, 0.2, 0.3, 0.5], ['0.06', '0.1', '0.2', '0.3', '0.5'])
plt.legend()
plt.show()
