import sys
sys.path.append("..")
sys.path.append(".")


from data.covariate import FactorModel
import numpy as np

lfm = FactorModel(p=1000, r=5, b_f=1, b_u=1)
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
cb = np.matmul(np.transpose(lfm.loadings), lfm.loadings)
eigen_values, eigen_vectors = largest_eigsh(cb, 5, which='LM')
print(eigen_values)

