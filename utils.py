import numpy as np


def unpack_loss(loss_set):
	loss_str = ""
	for name, value in loss_set.items():
		loss_str += f"{name}: {value} "
	return loss_str
