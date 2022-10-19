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
	(208 / 255.0, 84 / 255.0, 54 / 255.0),  # red
	(236 / 255.0, 129 / 255.0, 83 / 255.0),  # orange
	(40 / 255.0, 95 / 255.0, 127 / 255.0),  # dark blue
]
cand_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
l2_loss_matrix_mn = np.zeros((len(cand_p), 3))

for i, p in enumerate(cand_p):
	results = []
	for s in range(200):
		try:
			results.append(genfromtxt(f"../logs/exp4-hcm0-m200/p{p}s{s}.csv", delimiter=','))
		except:
			print(f"Load Data Error: no record p = {p}, s = {s}")
	result = np.array(results)
	l2_loss_matrix_mn[i, :] = np.mean(result, axis=0)


lines = [
	'dashed',
	'solid',
	'dashed',
]

model_name = [
	'Oracle-NN',
	'FAST-NN',
	'Oracle-Factor-NN',
]

plt.figure(figsize=(6, 6))
for i in range(3):
	plt.plot(cand_p, l2_loss_matrix_mn[:, i], color=color_tuple[i], linestyle=lines[i], label=model_name[i])

plt.ylabel(r"$\widehat{\mathtt{MSE}}$")
plt.xlabel(r"ambient dimension $p$")

plt.yscale("log")
plt.xscale("log")
# setting: HCM 0
plt.ylim([0.01, 5])
plt.yticks([0.02, 0.05, 0.12, 1.6], ['0.02', '0.05', '0.12', '1.6'])
# setting: HCM 3
# plt.ylim([0.03, 5])
# plt.yticks([0.05, 0.15, 0.2, 0.3, 1.5], ['0.05', '0.15', '0.2', '0.3', '1.5'])
plt.legend()
plt.show()
