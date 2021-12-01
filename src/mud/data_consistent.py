import pdb
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde


class DataConsistentProblem(object):
    """Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model paremeters
    using observed data and predicted data from the model.

    Parameters
    ----------
    X : ndarray
        2D array containing parameter samples from an initial distribution. Rows
        represent each sample while columns represent parameter values.
    y : ndarray
        array containing push-forward values of paramters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : array_like, optional
        2D Array containing ranges of each paramter value in the parameter
        space. Note that the number of rows must equal the number of parameters,
        and the number of columns must always be two, for min/max range.

    Example Usage
    -------------
    Generate test 1-D parameter estimation problem. Model to produce predicted
    data is the identity map and observed signal comes from true value plus
    some random gaussian nose:

    >>> from mud.base import DataConsistentProblem
    >>> from mud.funs import wme
    >>> import numpy as np
    >>> def test_wme_data(domain, num_samples, num_obs, noise, true):
    ...     # Parameter samples come from uniform distribution over domain
    ...     X = np.random.uniform(domain[0], domain[1], [num_samples,1])
    ...     # Identity map model, so predicted values same as param values.
    ...     predicted = np.repeat(X, num_obs, 1)
    ...     # Take Observed data from true value plus random gaussian noise
    ...     observed = np.ones(num_obs)*true + np.random.randn(num_obs)*noise
    ...     # Compute weighted mean error between predicted and observed values
    ...     y = wme(predicted, observed)
    ...     # Build density problem, with wme values as the model data
    ...     return DataConsistentProblem(X, y, [domain])

    Set up well-posed problem:
    >>> D = test_wme_data([0,1], 1000, 50, 0.05, 0.5)

    Estimate mud_point -> Note since WME map used, observed implied to be the
    standard normal distribution and does not have to be set explicitly from
    observed data set.
    >>> np.round(D.mud_point()[0],1)
    0.5

    Expecation value of r, ratio of observed and predicted distribution, should
    be near 1 if predictabiltiy assumption is satisfied.
    >>> np.round(D.exp_r(),0)
    1

    Set up ill-posed problem -> Searching out of range of true value
    >>> D = test_wme_data([0.6, 1], 1000, 50, 0.05, 0.5)

    Mud point will be close as we can get within the range we are searching for
    >>> np.round(D.mud_point()[0],1)
    0.6

    Expectation of r is close to zero since predictability assumption violated.
    >>> np.round(D.exp_r(),1)
    0.0

    """

    def __init__(self, X, y, domain=None, weights=None):

        # Set inputs
        self.X = X
        self.y = y
        self.domain = domain

        if self.y.ndim == 1:
            # Reshape 1D to 2D array to keep things consistent
            self.y = self.y.reshape(-1, 1)

        # Get dimensions of inverse problem
        self.param_dim = self.X.shape[1]
        self.obs_dim = self.y.shape[1]


        if self.domain is not None:
            # Assert our domain passed in is consistent with data array
            assert domain.shape[0]==self.X.shape[1]

        # Initialize distributions and descerte values to None
        self._r = None
        self._in = None
        self._pr = None
        self._ob = None
        self._up = None
        self._in_dist = None
        self._pr_dist = None
        self._ob_dist = None
        self._up_dist = None

        # Initialize Weights
        self.set_weights(weights)


    def set_weights(self, weights=None):
        if weights is None:
            w = np.ones(self.X.shape[0])
        else:
            w = weights.reshape(1, -1) if weights.ndim==1 else weights

            # Verify length of each weight vectors match number of samples in X
            assert self.X.shape[0]==w.shape[1]

            # Multiply weights column wise to get one weight row vector
            w = np.prod(w, axis=0)

            # Normalize weight vector
            w  = np.divide(w, np.sum(w,axis=0))

        # Re-set predicted, and updated since they're affected by weights
        self._weights = w
        self._pr = None
        self._up = None


    def set_observed(self, distribution=dist.norm()):
        """Set distribution for the observed data.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=scipy.stats.norm()
            scipy.stats continuous distribution like object representing the
            likelihood of observed data. Defaults to a standard normal
            distribution N(0,1).

        """
        self._ob_dist = distribution
        self._ob = distribution.pdf(self.y).prod(axis=1)


    def set_initial(self, distribution=None):
        """Set initial distribution of model parameter values.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, optional
            scipy.stats continuous distribution object from where initial
            parameter samples were drawn from. If non provided, then a uniform
            distribution over domain of density problem is assumed. If no domain
            is specified for density, then a standard normal distribution is
            assumed.

        """
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()
        self._in_dist = distribution
        self._in = self._in_dist.pdf(self.X).prod(axis=1) * self._weights
        self._up = None
        self._pr = None


    def set_predicted(self, distribution=None,
            bw_method=None, weights=None, **kwargs):
        """Sets the predicted distribution from predicted data `y`.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=None
            A scipy.stats continuous probability distribution. If non specified,
            then the distribution for the predicted data is computed using
            gaussina kernel density estimation.
        bw_method : str, scalar, or callable, optional
            Bandwidth method to use in gaussian kernel density estimation.
        weights : array_like, optional
            Weights to apply to predicted samples `y` in gaussian kernel density
            estimation.
        **kwargs : dict, optional
            If a distribution is passed, then any extra keyword arguments will
            be passed to the pdf() method as keyword arguments.

        Returns
        -------
        """
        if weights is not None:
            self.set_weights(weights)

        if distribution is None:
            distribution = gkde(self.y.T, bw_method=bw_method,
                    weights=self._weights)
            pred_pdf = distribution.pdf(self.y.T).T
        else:
            pred_pdf = distribution.pdf(self.y, **kwargs)
        self._pr_dist = distribution
        self._pr = pred_pdf
        self._up = None


    def fit(self):
        """Update initial distribution using ratio of observed and predicted.

        Applies [] to compute the updated distribution using the ratio of the
        observed to the predicted multiplied by the initial according to the
        data-consistent framework. Note that if initail, predicted, and observed
        distributiosn have not been set before running this method, they will
        be run with default values. To set specific predicted, observed, or
        initial distributions use the `set_` methods.

        Parameteres
        -----------

        Returns
        -----------

        """
        if self._in is None:
            self.set_initial()
        if self._pr is None:
            self.set_predicted()
        if self._ob is None:
            self.set_observed()

        # Store ratio of observed/predicted
        self._r = np.divide(self._ob, self._pr)

        # Compute only where observed is non-zero: NaN -> 0/0 -> set to 0.0
        self._r[np.argwhere(np.isnan(self._r))] = 0.0

        # Multiply by initial to get updated pdf
        self._up = np.multiply(self._in, self._r)


    def mud_point(self):
        """Maximal Updated Density (MUD) Point

        Returns the Maximal Updated Density or MUD point as the parameter sample
        from the initial distribution with the highest update density value.

        Parameters
        ----------

        Returns
        -------
        mud_point : ndarray
            Maximal Updated Density (MUD) point.
        """
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]


    def estimate(self):
        """Estimate

        Returns the best estimate for most likely paramter values for the given
        model data using the data-consistent framework.

        Parameters
        ----------

        Returns
        -------
        mud_point : ndarray
            Maximal Updated Density (MUD) point.
        """
        return self.mud_point()


    def exp_r(self):
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to the
        predicted density values. If the predictability assumption for the data-
        consistent framework is satisfied, then this value should be close to 1
        up to sampling errors.

        Parameters
        ----------

        Returns
        -------
        exp_r : float
            Value of the E(r). Should be close to 1.0.
        """
        if self._up is None:
            self.fit()

        return np.average(self._r, weights=self._weights)


    def plot_param_space(self,
            param_idx=0,
            ax=None,
            x_range=None,
            aff=1000,
            in_opts = {'color':'b', 'linestyle':'--', 'linewidth':4},
            up_opts = {'color':'k', 'linestyle':'-.', 'linewidth':4}):
        """
        Plot probability distributions over parameter space

        """

        if ax is None:
            _, ax = plt.subplots(1, 1)


        # Default x_range to full domain of all parameters
        x_range = x_range if x_range is not None else self.domain
        x_plot = np.linspace(x_range.T[0], x_range.T[1], num=aff)

        if in_opts is not None:
            # Compute initial plot based off of stored initial distribution
            in_plot = self._in_dist.pdf(x_plot)

            # Plot initial distribution over parameter space
            ax.plot(x_plot[:,param_idx], in_plot[:,param_idx], **in_opts)


        if up_opts is not None:
            # Compute r ratio if hasn't been already.
            if self._r is None:
                self.fit()

            # pi_up - kde over params weighted by r times previous weights
            up_plot = gkde(self.X.T, weights=self._r * self._weights)(x_plot.T)
            if self.param_dim==1:
                # Reshape two two-dimensional array if one-dim output
                up_plot = up_plot.reshape(-1,1)

            # Plut updated distribution over parameter space
            ax.plot(x_plot[:,param_idx], up_plot[:,param_idx], **up_opts)


    def plot_obs_space(self,
            obs_idx=0,
            ax=None,
            y_range=None,
            aff=1000,
            pf_in_opts = {'color':'b', 'linestyle':'--', 'linewidth':4},
            pf_up_opts = {'color':'k', 'linestyle':'-.', 'linewidth':4}):
        """
        Plot probability distributions over parameter space
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Default range is (-1,1) over each observable variable
        if y_range is None:
            y_range = np.repeat([[-1,1]], self.y.shape[1], axis=0)

        # Default x_range to full domain of all parameters
        y_plot = np.linspace(y_range.T[0], y_range.T[1], num=aff)

        if pf_in_opts is not None:
            # Compute PF of initial
            pf_in_plot = self._pr_dist(y_plot.T)
            if self.obs_dim==1:
                # Reshape two two-dimensional array if one-dim output
                pf_in_plot = pf_in_plot.reshape(-1,1)

            # Plot pf of initial
            ax.plot(y_plot[:,obs_idx], pf_in_plot[:,obs_idx], **pf_in_opts)

        if pf_up_opts is not None:
            # Compute r ratio if hasn't been already.
            if self._r is None:
                self.fit()

            # Compute PF of updated
            pf_up_plot = gkde(self.y.T, weights=self._r)(y_plot.T)
            if self.obs_dim==1:
                # Reshape two two-dimensional array if one-dim output
                pf_up_plot = pf_up_plot.reshape(-1,1)

            # Plut pf of updated
            ax.plot(y_plot[:,obs_idx], pf_up_plot[:,obs_idx], **pf_up_opts)

