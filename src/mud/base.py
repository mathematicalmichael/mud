import numpy as np
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde


class DensityProblem(object):
    def __init__(self, X, y, domain=None):
        self.X = X
        self.y = y
        self.domain = np.array(domain)
        self._up = None
        self._pr = None
        self._in = None
        self._ob = None

    def set_observed(self, distribution=dist.norm()):
        self._ob = distribution.pdf(self.y).prod(axis=1)

    def set_initial(self, distribution=None):
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            distribution = dist.norm()
        initial_dist = distribution
        self._in = initial_dist.pdf(self.X).prod(axis=1)

    def set_predicted(self, distribution=None):
        if distribution is None:
            distribution = gkde(self.y.T)
            pred_pdf = distribution.pdf(self.y.T).T
        else:
            pred_pdf = distribution.pdf(self.y)
        self._pr = pred_pdf

    def fit(self):
        if not self._in:
            self.set_initial()
            self._pr = None
        if not self._pr:
            self.set_predicted()
        if not self._ob:
            self.set_observed()

        up_pdf = np.divide(np.multiply(self._in, self._ob), self._pr)
        self._up = up_pdf

    def mud_point(self):
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]
