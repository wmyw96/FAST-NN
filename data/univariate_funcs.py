import numpy as np


def func1(x):
	return np.sin(x)


def func2(x):
	return np.sqrt(np.abs(x)) * 2 - 1


def func3(x):
	return (1 - np.abs(x)) ** 2


def func4(x):
	return 1 / (1.0 + np.exp(-x))


def func5(x):
	return np.cos(np.pi * x)
