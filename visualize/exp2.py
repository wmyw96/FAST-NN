import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=15)
rc('text', usetex=True)

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
]

cand_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
l2_loss_matrix_mn = np.zeros((len(cand_p), 4))
l2_loss_matrix_std = np.zeros((len(cand_p), 4))

for i, p in enumerate(cand_p):
	v = []
	exp1_result = None
	for k in ['0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.3']:
		results = []
		for s in range(200):
			try:
				results.append(genfromtxt(f"../logs/exp2-{k}/p{p}s{s}.csv", delimiter=','))
			except:
				print(f"Load Data Error: no record rate = {k}, p = {p}, s = {s}")
		ts = np.mean(np.array(results), 0)
		if k == '0':
			tmp = ts[2] + 0.0
			ts[2] = ts[3]
			ts[3] = tmp
			exp1_result = [ts[0], ts[1]]
		else:
			tmp = np.zeros((4,))
			tmp[2] = ts[0]
			tmp[3] = ts[1]
			tmp[0], tmp[1] = exp1_result[0], exp1_result[1]
			ts = tmp
		v.append(ts)

	v = np.array(v)
	l2_loss_matrix_mn[i, :] = np.min(v, 0)

model_name = [
	'Oracle-NN',
	'FAR-NN',
	'Dropout-NN-Joint',
	'Dropout-NN',
]

lines = [
	'dashed',
	'solid',
	'solid',
	'solid'
]

markers=[
	',',
	'8',
	'^',
	'v'
]


plt.figure(figsize=(6, 6))
for i in range(4):
	plt.plot(cand_p, l2_loss_matrix_mn[:, i], color=color_tuple[i], label=model_name[i], linestyle=lines[i],
			 marker=markers[i])

plt.ylabel(r"$\widehat{\mathtt{MSE}}$")
plt.xlabel(r"ambient dimension $p$")

plt.yscale("log")
plt.ylim([0.06, 0.75])
plt.yticks([0.08, 0.1, 0.2, 0.3, 0.5], ['0.08', '0.1', '0.2', '0.3', '0.5'])

plt.xscale("log")
plt.legend()
plt.show()
