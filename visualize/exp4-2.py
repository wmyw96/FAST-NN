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
	'#e5a84b',
	'#05348b',  # dark blue
]
cand_p = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
l2_loss_matrix_mn = np.zeros((len(cand_p), 4))
hcm_model = 3

for i, p in enumerate(cand_p):
	results = []
	for s in range(200):
		try:
			results.append(genfromtxt(f"../logs/exp4-hcm{hcm_model}-m200/p{p}s{s}.csv", delimiter=','))
		except:
			print(f"Load Data Error: no record p = {p}, s = {s}")
	result = np.array(results)
	l2_loss_matrix_mn[i, :] = np.mean(result, axis=0)


lines = [
	'dashed',
	'solid',
	'solid',
	'dashed',
]

model_name = [
	'Oracle-NN',
	'FAST-NN',
	'FAR-NN',
	'Oracle-Factor-NN',
]

marker = [
	',',
	'x',
	'8',
	','
]

plt.figure(figsize=(6, 6))
for i in range(4):
	plt.plot(cand_p, l2_loss_matrix_mn[:, i], color=color_tuple[i], linestyle=lines[i],
			 label=model_name[i], marker=marker[i])

#plt.plot([1, 8000], [0.10, 0.10], color='black', linestyle='dotted')

plt.ylabel(r"$\widehat{\mathtt{MSE}}$")
plt.xlabel(r"ambient dimension $p$")

plt.yscale("log")
plt.xscale("log")

if hcm_model == 0:
	# setting: HCM 0
	plt.ylim([0.005, 3])
	plt.yticks([0.01, 0.02, 0.12, 1.6], ['0.01', '0.02', '0.12', '1.6'])
else:

	# setting: HCM 3
	plt.ylim([0.02, 2.4])
	plt.yticks([0.05, 0.10, 0.25, 1.5], ['0.05', '0.10', '0.25', '1.5'])

plt.legend()
plt.show()
