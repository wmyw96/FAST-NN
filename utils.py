import numpy as np
import matplotlib.pyplot as plt

def unpack_loss(loss_set):
	loss_str = ""
	for name, value in loss_set.items():
		loss_str += f"{name}: {value} "
	return loss_str


def get_index_array(l, r):
	index = []
	for i in range(r - l + 1):
		index.append(i + l)


def visualize_matrix(mat):
	variable_selection_mat = np.abs(mat)
	row_sum = np.max(variable_selection_mat, axis=1)
	col_sum = np.max(variable_selection_mat, axis=0)
	print(col_sum)
	sorted_row = np.flip(np.argsort(row_sum))[:10]
	# sorted_col = np.flip(np.argsort(col_sum))[:50]
	# print(sorted_col)
	print(sorted_row)
	small_mat = np.zeros((10, 40))
	for i in range(10):
		for j in range(40):
			small_mat[i, j] = variable_selection_mat[i, j]

	from matplotlib import rc
	plt.figure(figsize=(12, 4))
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rc('font', size=15)
	rc('text', usetex=True)

	ax = plt.gca()
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	im = ax.imshow(np.log(small_mat), cmap="Reds")
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.15)
	cbar = ax.figure.colorbar(im, cax=cax)
	cbar.ax.tick_params(axis='both', which='major', labelsize=10)
	cbar.ax.tick_params(axis='both', which='minor', labelsize=10)
	cbar.ax.set_ylabel(r"$\log|[\hat{\Theta}^\top]_{i,j}|$", rotation=-90, va="bottom", fontsize=15)
	ax.set_xticks(np.arange(40), np.arange(40) + 1, fontsize=10)
	ax.set_yticks(np.arange(10), np.arange(10) + 1, fontsize=10)
	ax.set_xlabel(r"column $j$")
	ax.set_ylabel(r"row $i$")

	plt.savefig('a.pdf', bbox_inches='tight', pad_inches=0.05)
	plt.show()
