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


def func6(x):
	return np.sin(np.pi * x)


def func7(x):
	return -np.sin(x)


def func8(x):
	return np.cos(2 * x)


def func9(x):
	return np.tan(x + 0.1)


def func10(x):
	return np.log(x + 1.5)