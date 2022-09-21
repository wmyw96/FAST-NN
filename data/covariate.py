import numpy as np


class FactorModel:
    """
        The data generating process of linear factor model

        ...

        Attributes
        ----------
        loadings : numpy.array
            [p, r] factor loading matrix

    """

    def __init__(self, p, r=5, b_f=1, b_u=1, loadings=None):
        """
            Parameters
            ----------
            p : int
                number of covariates
            r : int
                number of factors
            b_f : float
                noise level of factors
            b_u : float
                noise level of idiosyncratic components
            loadings : numpy.array
                pre-specified factor loading matrix

            Returns
            -------
            loadings : numpy.array
                [p, r] matrix, factor loadings
        """

        self.p = p
        self.r = r
        self.b_f = b_f
        self.b_u = b_u
        if r > 0:
            if loadings is None:
                self.loadings = np.reshape(np.random.uniform(-np.sqrt(3), np.sqrt(3), p * r), (p, r))
            else:
                self.loadings = loadings
        else:
            self.loadings=None

    def sample(self, n, latent=False):
        """
            Parameters
            ----------
            n : int
                number of samples
            latent : bool
                whether return the latent factor structure

            Returns
            -------
            obs : np.array
                [n, p] matrix, observations
            factor : np.array
                [n, r] matrix, factor
            idiosyncratic_error : np.array
                [n, p] matrix, idiosyncratic error
        """
        if self.r > 0:
            factor = np.reshape(np.random.uniform(-self.b_f, self.b_f, n * self.r), (n, self.r))
        idiosyncratic_error = np.reshape(np.random.uniform(-self.b_u, self.b_u, self.p * n), (n, self.p))
        if self.r > 0:
            obs = np.matmul(factor, np.transpose(self.loadings)) + idiosyncratic_error
        else:
            obs = idiosyncratic_error
        if latent and self.r > 0:
            return obs, factor, idiosyncratic_error
        else:
            return obs
