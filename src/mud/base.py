import numpy as np
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde


class DensityProblem(object):
    """
    Sets up Data-Consistent Inverse Problem for parameter identification


    Example Usage
    -------------

    >>> from mud.base import DensityProblem
    >>> from mud.funs import wme
    >>> import numpy as np
    >>> X = np.random.rand(100,1)
    >>> num_obs = 50
    >>> Y = np.repeat(X, num_obs, 1)
    >>> y = np.ones(num_obs)*0.5 + np.random.randn(num_obs)*0.05
    >>> W = wme(Y, y)
    >>> B = DensityProblem(X, W, np.array([[0,1], [0,1]]))
    >>> np.round(B.mud_point()[0],1)
    0.5

    """

    def __init__(self, X, y, domain=None):
        self.X = X
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.domain = domain
        self._up = None
        self._in = None
        self._pr = None
        self._ob = None

    def set_observed(self, distribution=dist.norm()):
        self._ob = distribution.pdf(self.y).prod(axis=1)

    def set_initial(self, distribution=None):
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()
        initial_dist = distribution
        self._in = initial_dist.pdf(self.X).prod(axis=1)
        self._up = None
        self._pr = None

    def set_predicted(self, distribution=None, **kwargs):
        if distribution is None:
            distribution = gkde(self.y.T, **kwargs)
            pred_pdf = distribution.pdf(self.y.T).T
        else:
            pred_pdf = distribution.pdf(self.y, **kwargs)
        self._pr = pred_pdf
        self._up = None

    def fit(self, **kwargs):
        if self._in is None:
            self.set_initial()
            self._pr = None
        if self._pr is None:
            self.set_predicted(**kwargs)
        if self._ob is None:
            self.set_observed()

        up_pdf = np.divide(np.multiply(self._in, self._ob), self._pr)
        self._up = up_pdf

    def mud_point(self):
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]

    def estimate(self):
        return self.mud_point()


class BayesProblem(object):
    """
    Sets up Bayesian Inverse Problem for parameter identification


    Example Usage
    -------------

    >>> from mud.base import BayesProblem
    >>> import numpy as np
    >>> from scipy.stats import distributions as ds
    >>> X = np.random.rand(100,1)
    >>> num_obs = 50
    >>> Y = np.repeat(X, num_obs, 1)
    >>> y = np.ones(num_obs)*0.5 + np.random.randn(num_obs)*0.05
    >>> B = BayesProblem(X, Y, np.array([[0,1], [0,1]]))
    >>> B.set_likelihood(ds.norm(loc=y, scale=0.05))
    >>> np.round(B.map_point()[0],1)
    0.5

    """

    def __init__(self, X, y, domain=None):
        self.X = X
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.domain = domain
        self._ps = None
        self._pr = None
        self._ll = None

    def set_likelihood(self, distribution, log=False):
        if log:
            self._log = True
            self._ll = distribution.logpdf(self.y).sum(axis=1)
            # below is an equivalent evaluation (demonstrating the expected symmetry)
            # std, mean = distribution.std(), distribution.mean()
            # self._ll = dist.norm(self.y, std).logpdf(mean).sum(axis=1)
        else:
            self._log = False
            self._ll = distribution.pdf(self.y).prod(axis=1)
            # equivalent
            # self._ll = dist.norm(self.y).pdf(distribution.mean())/distribution.std()
            # self._ll = self._ll.prod(axis=1)
        self._ps = None

    def set_prior(self, distribution=None):
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()
        prior_dist = distribution
        self._pr = prior_dist.pdf(self.X).prod(axis=1)
        self._ps = None

    def fit(self):
        if self._pr is None:
            self.set_prior()
        if self._ll is None:
            self.set_likelihood()

        if self._log:
            ps_pdf = np.add(np.log(self._pr), self._ll)
        else:
            ps_pdf = np.multiply(self._pr, self._ll)

        assert ps_pdf.shape[0] == self.X.shape[0]
        if np.sum(ps_pdf) == 0:
            raise ValueError("Posterior numerically unstable.")
        self._ps = ps_pdf

    def map_point(self):
        if self._ps is None:
            self.fit()
        m = np.argmax(self._ps)
        return self.X[m, :]

    def estimate(self):
        return self.map_point()
