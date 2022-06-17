from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde
from mud.util import make_2d_unit_mesh, null_space


class DensityProblem(object):
    """
    Sets up Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model paremeters
    using observed data and predicted data from the model.

    Attributes
    ----------
    X : ArrayLike
        Array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
        If 1 dimensional input is passed, assumed that it represents repeated
        samples of a 1-dimensional parameter.
    y : ArrayLike
        Array containing push-forward values of paramters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : ArrayLike
        Array containing ranges of each paramter value in the parameter
        space. Note that the number of rows must equal the number of
        parameters, and the number of columns must always be two, for min/max
        range.
    weights : ArrayLike, optional
        Weights to apply to each parameter sample. Either a 1D array of the
        same length as number of samples or a 2D array if more than
        one set of weights is to be incorporated. If so the weights will be
        multiplied and normalized row-wise, so the number of columns must
        match the number of samples.

    Examples
    -------------

    Generate test 1-D parameter estimation problem. Model to produce predicted
    data is the identity map and observed signal comes from true value plus
    some random gaussian nose.

    See :meth:`mud.examples.identity_uniform_1D_density_prob` for more details

    >>> from mud.examples import identity_uniform_1D_density_prob as I1D

    First we set up a well-posed problem. Note the domain we are looking over
    contains our true value. We take 1000 samples, use 50 observations,
    assuming a true value of 0.5 populated with gaussian noise
    :math:`\\mathcal{N}(0,0.5)`. Or initial uniform distribution is taken from a
    :math:`[0,1]` range.

    >>> D = I1D(1000, 50, 0.5, 0.05, domain=[0,1])

    Estimate mud_point -> Note since WME map used, observed implied to be the
    standard normal distribution and does not have to be set explicitly from
    observed data set.

    >>> np.round(D.mud_point()[0],1)
    0.5

    Expecation value of r, ratio of observed and predicted distribution, should
    be near 1 if predictabiltiy assumption is satisfied.

    >>> np.round(D.expected_ratio(),0)
    1.0

    Set up ill-posed problem -> Searching out of range of true value

    >>> D = I1D(1000, 50, 0.5, 0.05, domain=[0.6,1])

    Mud point will be close as we can get within the range we are searching for

    >>> np.round(D.mud_point()[0],1)
    0.6

    Expectation of r is close to zero since predictability assumption violated.

    >>> np.round(D.expected_ratio(),1)
    0.0

    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        domain: Union[np.ndarray, List] = None,
        weights: Union[np.ndarray, List] = None,
    ):

        # Set and validate inputs. Note we reshape inputs as necessary
        def set_shape(x, y):
            return x.reshape(y) if x.ndim < 2 else x
        self.X = set_shape(np.array(X), (1, -1))
        self.y = set_shape(np.array(y), (-1, 1))
        self.domain = set_shape(np.array(domain), (1, -1))

        # These will be updated in set_ and fit() functions
        self._r = None  # Ratio of observed to predicted
        self._up = None  # Updated values
        self._in = None  # Initial values
        self._pr = None  # Predicted values
        self._ob = None  # Observed values
        self._in_dist = None  # Initial distirbution
        self._pr_dist = None  # Predicted distribution
        self._ob_dist = None  # Observed distribution

        if self.domain is not None:
            # Assert domain passed in is consitent with data array
            assert self.domain.shape[0] == self.n_params

        # Iniitialize weights
        self.set_weights(weights)

    @property
    def n_params(self):
        return self.X.shape[1]

    @property
    def n_features(self):
        return self.y.shape[1]

    @property
    def n_samples(self):
        return self.y.shape[0]

    def set_weights(self, weights: Union[np.ndarray, List], normalize: bool = False):
        """Set Sample Weights

        Sets the weights to use for each sample. Note weights can be one or two
        dimensional. If weights are two dimensional the weights are combined
        by multiplying them row wise and normalizing, to give one weight per
        sample. This combining of weights allows incorporating multiple sets
        of weights from different sources of prior belief.

        Parameters
        ----------
        weights : np.ndarray, List
            Numpy array or list of same length as the `n_samples` or if two
            dimensional, number of columns should match `n_samples`
        normalise : bool, default=False
            Whether to normalize the weights vector.

        Returns
        -------

        Warnings
        --------
        Resetting weights will delete the predicted and updated distirbution
        values in the class, requiring a re-run of adequate `set_` methods
        and/or `fit()` to reproduce with new weights.
        """
        if weights is None:
            w = np.ones(self.X.shape[0])
        else:
            if isinstance(weights, list):
                weights = np.array(weights)

            # Reshape to 2D
            w = weights.reshape(1, -1) if weights.ndim == 1 else weights

            # assert appropriate size
            assert self.n_samples == w.shape[1], f"`weights` must size {self.n_samples}"

            # Multiply weights column wise for stacked weights
            w = np.prod(w, axis=0)

            # Normalize weight vector
            if normalize:
                w = np.divide(w, np.sum(w, axis=0))

        self._weights = w
        self._pr = None
        self._up = None
        self._pr_dist = None

    def set_observed(self, distribution: rv_continuous = dist.norm()):
        """Set distribution for the observed data.

        The observed distribution is determined from assumptions on the
        collected data. In the case of using a weighted mean error map on
        sequential data from a single output, the distribution is stationary
        with respect to the number data points collected and will always be
        the standard normal d distribution $N(0,1)$.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=scipy.stats.norm()
            scipy.stats continuous distribution like object representing the
            likelihood of observed data. Defaults to a standard normal
            distribution N(0,1).

        """
        self._ob_dist = distribution
        self._ob = distribution.pdf(self.y).prod(axis=1)

    def set_initial(self, distribution: rv_continuous = None):
        """
        Set initial probability distribution of model parameter values
        :math:`\\pi_{in}(\\lambda)`.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, optional
            scipy.stats continuous distribution object from where initial
            parameter samples were drawn from. If none provided, then a uniform
            distribution over domain of the density problem is assumed. If no
            domain is specified for density, then a standard normal
            distribution :math:`N(0,1)` is assumed.

        Warnings
        --------
        Setting initial distirbution resets the predicted and updated
        distributions, so make sure to set the initial first.
        """
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()

        self._in_dist = distribution
        self._in = self._in_dist.pdf(self.X).prod(axis=1)
        self._up = None
        self._pr = None
        self._pr_dist = None

    def set_predicted(
        self,
        distribution: rv_continuous = None,
        bw_method: Union[str, callable, np.generic] = None,
        weights: ArrayLike = None,
        **kwargs,
    ):
        """
        Set Predicted Distribution

        The predicted distribution over the observable space is equal to the
        push-forward of the initial through the model
        :math:`\\pi_{pr}(Q(\\lambda)`. If no distribution is passed,
        :class:`scipy.stats.gaussian_kde` is used over the predicted values
        :attr:`y` to estimate the predicted distribution.

        Parameters
        ----------
        distribution : :class:`scipy.stats.rv_continuous`, optional
            If specified, used as the predicted distribution instead of the
            default of using gaussian kernel density estimation on observed
            values y. This should be a frozen distribution if using
            `scipy`, and otherwise be a class containing a `pdf()` method
            return the probability density value for an array of values.
        bw_method : str, scalar, or callable, optional
            Method to use to calculate estimator bandwidth. Only used if
            distribution is not specified, See documentation for
            :class:`scipy.stats.gaussian_kde` for more information.
        weights : ArrayLike, optional
            Weights to use on predicted samples. Note that if specified,
            :meth:`set_weights` will be run first to calculate new weights.
            Otherwise, whatever was previously set as the weights is used.
            Note this defaults to a weights vector of all 1s for every sample
            in the case that no weights were passed on upon initialization.
        **kwargs: dict, optional
            If specified, any exra keyword arguments will be passed along to
            the passed ``distribution.pdf()`` function for computing values of
            predicted samples.

        Note: `distribution` should be a frozen distribution if using `scipy`.

        Warnings
        --------
        If passing a `distribution` argument, make sure that the initial
        distribution has been set first, either by having run
        :meth:`set_initial` or :meth:`fit` first.
        """
        if weights is not None:
            self.set_weights(weights)

        if distribution is None:
            # Reweight kde of predicted by weights if present
            distribution = gkde(self.y.T, bw_method=bw_method, weights=self._weights)
            pred_pdf_values = distribution.pdf(self.y.T).T
        else:
            pred_pdf_values = distribution.pdf(self.y, **kwargs)

        self._pr_dist = distribution
        self._pr = pred_pdf_values.ravel()
        self._up = None

    def fit(self, **kwargs):
        """
        Update Initial Distribution

        Constructs the updated distribution by fiting osberved data to
        predicted data with:

        .. math::
            \\pi_{up}(\\lambda) = \\pi_{in}(\\lambda)
            \\frac{\\pi_{ob}(Q(\\lambda))}{\\pi_{pred}(Q(\\lambda))}
            :label: data_consistent_solution

        Note that if initial, predicted, and observed distributions have not
        been set before running this method, they will be run with default
        values. To set specific predicted, observed, or initial distributions
        use the ``set_`` methods.

        Parameters
        -----------
        **kwargs : dict, optional
            If specified, optional arguments are passed to the
            :meth:`set_predicted` call in the case that the predicted
            distribution has not been set yet.

        Returns
        -----------

        """
        if self._in is None:
            self.set_initial()
        if self._pr is None:
            self.set_predicted(**kwargs)
        if self._ob is None:
            self.set_observed()

        # Store ratio of observed/predicted
        # e.g. to comptue E(r) and to pass on to future iterations
        self._r = np.divide(self._ob, self._pr)

        # Multiply by initial to get updated pdf
        up_pdf = np.multiply(self._in * self._weights, self._r)
        self._up = up_pdf

    def mud_point(self):
        """Maximal Updated Density (MUD) Point

        Returns the Maximal Updated Density or MUD point as the parameter
        sample from the initial distribution with the highest update density
        value:

        .. math::
            \\lambda^{MUD} := \\text{argmax} \\pi_{up}(\\lambda)
            :label: mud

        Note if the updated distribution has not been computed yet, this
        function will call :meth:`fit` to compute it.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
            Maximal Updated Density (MUD) point.
        """
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]

    def estimate(self):
        """Estimate

        Returns the best estimate for most likely paramter values for the
        given model data using the data-consistent framework.

        Parameters
        ----------

        Returns
        -------
        mud_point : ndarray
            Maximal Updated Density (MUD) point.
        """
        return self.mud_point()

    def expected_ratio(self):
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to
        the predicted density values.

        .. math::
            R = \\frac{\\pi_{ob}(\\lambda)}
                      {\\pi_{pred}(\\lambda)}
            :label: r_ratio

        If the predictability assumption for the data-consistent framework is
        satisfied, then :math:`E[R]\\approx 1`.

        Parameters
        ----------

        Returns
        -------
        expected_ratio : float
            Value of the E(r). Should be close to 1.0.
        """
        if self._up is None:
            self.fit()

        return np.average(self._r, weights=self._weights)

    def plot_param_space(
        self,
        param_idx: int = 0,
        ax: plt.Axes = None,
        x_range: Union[list, np.ndarray] = None,
        aff: int = 1000,
        in_opts={"color": "b", "linestyle": "--", "linewidth": 4, "label": "Initial"},
        up_opts={"color": "k", "linestyle": "-.", "linewidth": 4, "label": "Updated"},
        win_opts={
            "color": "g",
            "linestyle": "--",
            "linewidth": 4,
            "label": "Weighted Initial",
        },
    ):
        """
        Plot probability distributions over parameter space

        Initial distribution is plotted using the distribution function passed
        to :meth:`set_initial`. The updated distribution is
        plotted using a weighted gaussian kernel density estimate (gkde) on the
        initial samples, using the product of the update ratio :eq:`r_ratio`
        value times the initial weights as weights for the gkde. The weighted
        initial is built using a weighted gkde on the initial samples, but
        only using the initial weights.

        Parameters
        ----------
        param_idx : int, default=0
            Index of parameter value to plot.
        ax : :class:`matplotlib.axes.Axes`, optional
            Axes to plot distributions on. If non specified, a figure will
            be initialized to plot on.
        x_range : list or np.ndarray, optional
            Range over parameter value to plot over.
        aff : int, default=100
            Number of points to plot within x_range, evenly spaced.
        in_opts : dict, optional
            Plotting option for initial distribution line. Defaults to
            ``{'color':'b', 'linestyle':'--','linewidth':4,
            'label':'Initial'}``. To supress plotting, pass in ``None``
            explicitly.
        up_opts : dict, optional
            Plotting option for updated distribution line. Defaults to
            ``{'color':'k', 'linestyle':'-.','linewidth':4,
            'label':'Updated'}``. To supress plotting, pass in ``None``
            explicitly.
        win_opts : dict, optional
            Plotting option for weighted initial distribution line. Defaults to
            ``{'color':'g', 'linestyle':'--','linewidth':4,
            'label':'Weighted Initial'}``. To supress plotting, pass in
            ``None`` explicitly.

        Returns
        -------
        """
        # Default options for plotting figures
        io = {"color": "b", "linestyle": "--", "linewidth": 4, "label": "Initial"}
        uo = {"color": "k", "linestyle": "-.", "linewidth": 4, "label": "Updated"}
        wo = {
            "color": "g",
            "linestyle": "--",
            "linewidth": 4,
            "label": "Weighted Initial",
        }

        # Create plot if one isn't passed in
        _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

        # Default x_range to full domain of all parameters
        x_range = x_range if x_range is not None else self.domain
        x_plot = np.linspace(x_range.T[0], x_range.T[1], num=aff)

        # Plot distributions for all not set to None
        if in_opts:
            # Update default options with passed in options
            io.update(in_opts)

            # Compute initial plot based off of stored initial distribution
            in_plot = self._in_dist.pdf(x_plot)
            in_plot = in_plot.reshape(-1, 1) if self.n_params == 1 else in_plot

            # Plot initial distribution over parameter space
            ax.plot(x_plot[:, param_idx], in_plot[:, param_idx], **io)
        if up_opts:
            # Update options with passed in options
            uo.update(up_opts)

            # pi_up - kde over params weighted by r times previous weights
            up_plot = gkde(self.X.T, weights=self._r * self._weights)(x_plot.T)
            up_plot = up_plot.reshape(-1, 1) if self.n_params == 1 else up_plot

            # Plut updated distribution over parameter space
            ax.plot(x_plot[:, param_idx], up_plot[:, param_idx], **uo)
        if win_opts:
            # Update default options with passed in options
            wo.update(win_opts)

            # Compute weighted initial based off of KDE initial samples
            w_plot = gkde(self.X[:, param_idx], weights=self._weights)(x_plot.T)
            w_plot = w_plot.reshape(-1, 1) if self.n_params == 1 else w_plot

            # Plot KDE estimate of weighted input distribution using samples
            ax.plot(x_plot[:, param_idx], w_plot[:, param_idx], **wo)

    def plot_obs_space(
        self,
        obs_idx: int = 0,
        ax: plt.Axes = None,
        y_range: ArrayLike = None,
        aff=1000,
        ob_opts={"color": "r", "linestyle": "-", "linewidth": 4, "label": "Observed"},
        pr_opts={
            "color": "b",
            "linestyle": "--",
            "linewidth": 4,
            "label": "PF of Initial",
        },
        pf_opts={
            "color": "k",
            "linestyle": "-.",
            "linewidth": 4,
            "label": "PF of Updated",
        },
    ):
        """
        Plot probability distributions over parameter space

        Observed distribution is plotted using the distribution function passed
        to :meth:`set_observed` (or defaul). The predicted distribution is
        plotted using the stored predicted distribution function set in
        :meth:`set_predicted`. The push-forward of the updated distribution is
        computed as a gkde on the predicted samples :attr:`y` as well, but
        using the product of the update ratio :eq:`r_ratio` and the initial
        weights as weights.

        Parameters
        ----------
        obs_idx: int, default=0
            Index of observable value to plot.
        ax : :class:`matplotlib.axes.Axes`, optional
            Axes to plot distributions on. If non specified, a figure will
            be initialized to plot on.
        y_range : list or np.ndarray, optional
            Range over parameter value to plot over.
        aff : int, default=100
            Number of points to plot within x_range, evenly spaced.
        ob_opts : dict, optional
            Plotting option for observed distribution line. Defaults to
            ``{'color':'r', 'linestyle':'-','linewidth':4,
            'label':'Observed'}``. To supress plotting, pass in ``None``.
        pr_opts : dict, optional
            Plotting option for predicted distribution line. Defaults to
            ``{'color':'b', 'linestyle':'--','linewidth':4,
            'label':'PF of Initial'}``. To supress plotting, pass in ``None``.
        pf_opts : dict, optional
            Plotting option for push-forward of updated destribution line.
            Defaults to ``{'color':'k', 'linestyle':'-.','linewidth':4,
            'label':'PF of Updated'}``. To supress plotting, pass in
            ``None``.

        Returns
        -------
        """
        # observed, predicted, and push-forward opts respectively
        oo = {"color": "r", "linestyle": "-", "linewidth": 4, "label": "Observed"}
        po = {"color": "b", "linestyle": "-.", "linewidth": 4, "label": "PF of Initial"}
        fo = {"color": "k", "linestyle": "-.", "linewidth": 4, "label": "PF of Updated"}

        # Create plot if one isn't passed in
        _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

        # Default range is (-1,1) over each observable variable
        # TODO: Infer range from predicted y vals
        if y_range is None:
            y_range = np.repeat([[-1, 1]], self.n_features, axis=0)
        y_plot = np.linspace(y_range.T[0], y_range.T[1], num=aff)

        if ob_opts:
            # Update options with passed in values
            oo.update(ob_opts)

            # Compute observed distribution using stored pdf
            ob_p = self._ob_dist.pdf(y_plot.T)
            ob_p = ob_p.reshape(-1, 1) if self.n_features == 1 else ob_p

            # Plot observed density
            ax.plot(y_plot[:, obs_idx], ob_p[:, obs_idx], **oo)

        if pr_opts:
            # Update options with passed in values
            po.update(pr_opts)

            # Compute PF of initial - Predicted
            pr_p = self._pr_dist.pdf(y_plot.T)
            pr_p = pr_p.reshape(-1, 1) if self.n_features == 1 else pr_p

            # Plot pf of initial
            ax.plot(y_plot[:, obs_idx], pr_p[:, obs_idx], **pr_opts)

        if pf_opts is not None:
            fo.update(pf_opts)

            # Compute PF of updated
            pf_p = gkde(self.y.T, weights=self._weights * self._r)(y_plot.T)
            pf_p = pf_p.reshape(-1, 1) if self.n_features == 1 else pf_p

            # Plut pf of updated
            ax.plot(y_plot[:, obs_idx], pf_p[:, obs_idx], **pf_opts)


class BayesProblem(object):
    """
    Sets up Bayesian Inverse Problem for parameter identification

    Parameters
    ----------
    X : ndarray
        2D array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
    y : ndarray
        array containing push-forward values of paramters samples through the
        forward model. These samples will form the data-likelihood
        distribution.
    domain : array_like, optional
        2D Array containing ranges of each paramter value in the parameter
        space. Note that the number of rows must equal the number of
        parameters, and the number of columns must always be two, for min/max
        range.

    Examples
    --------

    >>> from mud.base import BayesProblem
    >>> import numpy as np
    >>> from scipy.stats import distributions as ds
    >>> X = np.random.rand(100,1)
    >>> num_obs = 50
    >>> Y = np.repeat(X, num_obs, 1)
    >>> y = np.ones(num_obs)*0.5 + np.random.randn(num_obs)*0.05
    >>> B = BayesProblem(X, Y, np.array([[0,1]]))
    >>> B.set_likelihood(ds.norm(loc=y, scale=0.05))
    >>> np.round(B.map_point()[0],1)
    0.5

    """

    def __init__(
        self,
        X: Union[np.ndarray, List],
        y: Union[np.ndarray, List],
        domain: Union[np.ndarray, List] = None,
    ):

        # Initialize inputs
        self.X = np.array(X)
        self.y = np.array(y)
        self.y = self.y.reshape(-1, 1) if y.ndim == 1 else y
        self.domain = np.array(domain).reshape(1, -1)

        if self.domain is not None:
            # Assert our domain passed in is consistent with data array
            assert self.domain.shape[0] == self.n_params

        # Initialize ps, predicted, and likelihood values/distributions
        self._ps = None
        self._pr = None
        self._ll = None
        self._ll_dist = None
        self._pr_dist = None

    @property
    def n_params(self):
        return self.X.shape[1]

    @property
    def n_features(self):
        return self.y.shape[1]

    @property
    def n_samples(self):
        return self.y.shape[0]

    def set_likelihood(self, distribution, log=False):
        self._ll_dist = distribution
        if log:
            self._log = True
            self._ll = distribution.logpdf(self.y).sum(axis=1)
            # equivalent evaluation (demonstrating the expected symmetry)
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
        self._pr_dist = distribution
        self._pr = self._pr_dist.pdf(self.X).prod(axis=1)
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

    def plot_param_space(
        self,
        param_idx=0,
        ax=None,
        x_range=None,
        aff=1000,
        pr_opts={"color": "b", "linestyle": "--", "linewidth": 4, "label": "Prior"},
        ps_opts={"color": "g", "linestyle": ":", "linewidth": 4, "label": "Posterior"},
    ):
        """
        Plot probability distributions over parameter space

        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Default x_range to full domain of all parameters
        x_range = x_range if x_range is not None else self.domain
        x_plot = np.linspace(x_range.T[0], x_range.T[1], num=aff)

        if pr_opts is not None:
            # Compute initial plot based off of stored initial distribution
            pr_plot = self._pr_dist.pdf(x_plot)

            # Plot prior distribution over parameter space
            ax.plot(x_plot[:, param_idx], pr_plot[:, param_idx], **pr_opts)

        if ps_opts is not None:
            # Compute posterior if it hasn't been already
            if self._ps is None:
                raise ValueError("posterior not set yet. Run fit()")

            # ps_plot - kde over params weighted by posterior computed pdf
            ps_plot = gkde(self.X.T, weights=self._ps)(x_plot.T)
            if self.n_params == 1:
                # Reshape two two-dimensional array if one-dim output
                ps_plot = ps_plot.reshape(-1, 1)

            # Plot posterior distribution over parameter space
            ax.plot(x_plot[:, param_idx], ps_plot[:, param_idx], **ps_opts)

    def plot_obs_space(
        self,
        obs_idx=0,
        ax=None,
        y_range=None,
        aff=1000,
        ll_opts={
            "color": "r",
            "linestyle": "-",
            "linewidth": 4,
            "label": "Data-Likelihood",
        },
        pf_opts={
            "color": "g",
            "linestyle": ":",
            "linewidth": 4,
            "label": "PF of Posterior",
        },
    ):
        """
        Plot probability distributions defined over observable space.
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Default range is (-1,1) over each observable variable
        if y_range is None:
            y_range = np.repeat([[-1, 1]], self.y.shape[1], axis=0)

        # Default x_range to full domain of all parameters
        y_plot = np.linspace(y_range.T[0], y_range.T[1], num=aff)

        if ll_opts is not None:
            if self._ll is None:
                raise ValueError("Likelihood not set. Run fit()")

            # Compute Likelihoood values
            ll_plot = self._ll_dist.pdf(y_plot).prod(axis=1)

            if self.n_features == 1:
                # Reshape two two-dimensional array if one-dim output
                ll_plot = ll_plot.reshape(-1, 1)

            # Plot pf of initial
            ax.plot(y_plot[:, obs_idx], ll_plot[:, obs_idx], **ll_opts)

        if pf_opts is not None:
            # Compute PF of updated
            pf_plot = gkde(self.y.T, weights=self._ps)(y_plot.T)
            if self.n_features == 1:
                # Reshape two two-dimensional array if one-dim output
                pf_plot = pf_plot.reshape(-1, 1)

            # Plut pf of updated
            ax.plot(y_plot[:, obs_idx], pf_plot[:, obs_idx], **pf_opts)


class LinearGaussianProblem(object):
    """Sets up inverse problems with Linear/Affine Maps

    Class provides solutions using MAP, MUD, and least squares solutions to the
    linear (or affine) problem from `p` parameters to `d` observables.

    .. math ::
        M(\\mathbf{x}) = A\\mathbf{x} + \\mathbf{b},
        A \\in \\mathbb{R}^{d\\times p},
        \\mathbf{x}, \\in \\mathbb{R}^{p},
        \\mathbf{b}, \\in \\mathbb{R}^{d},
        :label: linear_map

    Attributes
    ----------
    A : ArrayLike
        2D Array defining kinear transformation from model parameter space to
        model output space.
    y : ArrayLike
        1D Array containing observed values of Q(\\lambda)
        Array containing push-forward values of paramters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : ArrayLike
        Array containing ranges of each paramter value in the parameter
        space. Note that the number of rows must equal the number of
        parameters, and the number of columns must always be two, for min/max
        range.
    weights : ArrayLike, optional
        Weights to apply to each parameter sample. Either a 1D array of the
        same length as number of samples or a 2D array if more than
        one set of weights is to be incorporated. If so the weights will be
        multiplied and normalized row-wise, so the number of columns must
        match the number of samples.

    Examples
    -------------

    Problem set-up:

    .. math ::
        A = \\begin{bmatrix} 1 & 1 \\end{bmatrix}, b = 0, y = 1
        \\lambda_0 = \\begin{bmatrix} 0.25 & 0.25 \\end{bmatrix}^T,
        \\Sigma_{init} = \\begin{bmatrix} 1 & -0.25 \\\\ -0.25 & 0.5 \\end{bmatrix},
        \\Sigma_{obs} = \\begin{bmatrix} 0.25 \\end{bmatrix}

    >>> from mud.base import LinearGaussianProblem as LGP
    >>> lg1 = LGP(A=np.array([[1, 1]]),
    ...        b=np.array([[0]]),
    ...        y=np.array([[1]]),
    ...        mean_i=np.array([[0.25, 0.25]]).T,
    ...        cov_i=np.array([[1, -0.25], [-0.25, 0.5]]),
    ...        cov_o=np.array([[1]]))
    >>> lg1.solve('mud')
    array([[0.625],
           [0.375]])

    """

    def __init__(
        self,
        A=np.array([1, 1]).reshape(-1, 1),
        b=None,
        y=None,
        mean_i=None,
        cov_i=None,
        cov_o=None,
        alpha=1.0,
    ):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)
        ns, di = self.A.shape

        # Initialize to defaults - Reshape everything into 2D arrays.
        self.b = np.zeros((ns, 1)) if b is None else b.reshape(-1, 1)
        self.y = np.zeros((ns, 1)) if y is None else y.reshape(-1, 1)
        self.mean_i = np.zeros((di, 1)) if mean_i is None else mean_i.reshape(-1, 1)
        self.cov_i = np.eye(di) if cov_i is None else cov_i
        self.cov_o = np.eye(ns) if cov_o is None else cov_o

        # How much to scale regularization terms
        self.alpha = alpha

        # Check appropriate dimensions of inputs
        n_data, n_targets = self.y.shape
        if ns != n_data:
            raise ValueError(
                "Number of samples in X and y does not correspond:"
                " %d != %d" % (ns, n_data)
            )

        # Initialize to no solution
        self.sol = None

    @property
    def n_params(self):
        return self.A.shape[1]

    @property
    def n_features(self):
        return self.y.shape[1]

    @property
    def n_samples(self):
        return self.y.shape[0]

    def compute_functionals(self, X, terms="all"):
        """
        For a given input and observed data, compute functionals or
        individual terms in functionals that are minimized to solve the
        linear gaussian problem.
        """
        # Compute observed mean
        mean_o = self.y - self.b

        # Define inner-producted induced by vector norm
        def ip(X, mat):
            return np.sum(X * (np.linalg.inv(mat) @ X), axis=0)

        # First compute data mismatch norm
        data_term = ip((self.A @ X.T + self.b) - mean_o.T, self.cov_o)
        if terms == "data":
            return data_term

        # Tikhonov Regularization Term
        reg_term = self.alpha * ip((X - self.mean_i.T).T, self.cov_i)
        if terms == "reg":
            return reg_term

        # Data-Consistent Term - "unregularizaiton" in data-informed directions
        dc_term = self.alpha * ip(
            self.A @ (X - self.mean_i.T).T, self.A @ self.cov_i @ self.A.T
        )
        if terms == "dc_term":
            return dc_term

        # Modified Regularization Term
        reg_m_terms = reg_term - dc_term
        if terms == "reg_m":
            return reg_m_terms

        bayes_fun = data_term + reg_term
        if terms == "bayes":
            return bayes_fun

        dc_fun = bayes_fun - dc_term
        if terms == "dc":
            return dc_fun

        return (data_term, reg_term, dc_term, bayes_fun, dc_fun)

    def solve(self, method="mud", output_dim=None):
        """
        Explicitly solve linear problem using given method.

        """
        # Reduce output dimension if desired
        od = self.A.shape[0] if output_dim is None else output_dim
        _A = self.A[:od, :]
        _b = self.b[:od, :]
        _y = self.y[:od, :]
        _cov_o = self.cov_o[:od, :od]

        # Compute observed mean
        mean_o = _y - _b

        # Compute residual
        z = mean_o - _A @ self.mean_i

        # Weight initial covariance to use according to alpha parameter
        a_cov_i = self.alpha * self.cov_i

        # Solve according to given method, or solve all methods
        if method == "mud" or method == "all":
            inv_pred_cov = np.linalg.pinv(_A @ a_cov_i @ _A.T)
            update = a_cov_i @ _A.T @ inv_pred_cov
            self.mud = self.mean_i + update @ z

        # if method == 'mud_alt' or method == 'all':
        #     up_cov = updated_cov(X=_A, init_cov=a_cov_i, data_cov=_cov_o)
        #     update = up_cov @ _A.T @ np.linalg.inv(_cov_o)
        #     self.mud_alt = self.mean_i + update @ z

        if method == "map" or method == "all":
            co_inv = np.linalg.inv(_cov_o)
            cov_p = np.linalg.inv(_A.T @ co_inv @ _A + np.linalg.inv(a_cov_i))
            update = cov_p @ _A.T @ co_inv
            self.map = self.mean_i + update @ z

        if method == "ls" or method == "all":
            # Compute ls solution from pinv method
            self.ls = np.linalg.pinv(_A) @ mean_o

        # Return solution or all solutions
        if method == "all":
            return (self.mud, self.map, self.ls)
            # return (self.mud, self.mud_alt, self.map, self.ls)
        else:
            return self.__getattribute__(method)

    def plot_sol(
        self,
        point="mud",
        ax=None,
        label=None,
        note_loc=None,
        pt_opts={"color": "k", "s": 100, "marker": "o"},
        ln_opts={"color": "xkcd:blue", "marker": "d", "lw": 1, "zorder": 10},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    ):
        """
        Plot solution points
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Get solution point or initial poitn to plot.
        pt = self.mean_i if point == "initial" else self.solve(method=point)
        pt_opts["label"] = point

        # Plot point
        ax.scatter(pt[0], pt[1], **pt_opts)

        # Plot line connecting iniital value and solution
        if ln_opts is not None and point != "initial":
            ax.plot(
                [self.mean_i.ravel()[0], pt.ravel()[0]],
                [self.mean_i.ravel()[1], pt.ravel()[1]],
                **ln_opts,
            )

        if label is not None:
            # Annotate point with a label if desired
            nc = note_loc
            nc = (pt[0] - 0.02, pt[1] + 0.02) if nc is None else nc
            ax.annotate(label, nc, **annotate_opts)

    def plot_contours(
        self,
        ref=None,
        subset=None,
        ax=None,
        annotate=False,
        note_loc=None,
        w=1,
        label="{i}",
        plot_opts={"color": "k", "ls": ":", "lw": 1, "fs": 20},
        annotate_opts={"fontsize": 20},
    ):
        """
        Plot Linear Map Solution Contours
        """
        # Initialize a plot if one hasn't been already
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # All rows of A are default subset of contours to plot
        subset = np.arange(self.A.shape[0]) if subset is None else subset

        # Ref is the reference point to plot each contour line through.
        ref = ref if ref is not None else self.solve(method="ls")

        # Build null-space (contour lines) for each subset row of A
        A = self.A[np.array(subset), :]
        numQoI = A.shape[0]
        AA = np.hstack([null_space(A[i, :].reshape(1, -1)) for i in range(numQoI)]).T

        # Plot each contour line going through ref point
        for i, contour in enumerate(subset):
            xloc = [ref[0] - w * AA[i, 0], ref[1] + w * AA[i, 0]]
            yloc = [ref[0] - w * AA[i, 1], ref[1] + w * AA[i, 1]]
            ax.plot(xloc, yloc, **plot_opts)

            # If annotate is set, then label line with given annotations
            if annotate:
                nl = (xloc[0], yloc[0]) if note_loc is None else note_loc
                ax.annotate(label.format(i=contour + 1), nl, **annotate_opts)

    def plot_fun_contours(self, mesh=None, terms="dc", ax=None, N=250, r=1, **kwargs):
        """
        Plot contour map offunctionals being minimized over input space
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Get mesh if one hasn't been passed
        if mesh is None:
            _, _, mesh = make_2d_unit_mesh(N, r)

        # Compute functional terms desired over range
        term = self.compute_functionals(mesh, terms=terms)

        # Plot contours
        _ = ax.contour(
            mesh[:, 0].reshape(N, N),
            mesh[:, 1].reshape(N, N),
            term.reshape(N, N),
            **kwargs,
        )


class IterativeLinearProblem(LinearGaussianProblem):
    def __init__(
        self, A, b, y=None, mu_i=None, cov=None, data_cov=None, idx_order=None
    ):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)

        # Initialize to defaults - Reshape everythin into 2D arrays.
        n_samples, dim_input = self.A.shape
        self.data_cov = np.eye(n_samples) if data_cov is None else data_cov
        self.cov = np.eye(dim_input) if cov is None else cov
        self.mu_i = np.zeros((dim_input, 1)) if mu_i is None else mu_i.reshape(-1, 1)
        self.b = np.zeros((n_samples, 1)) if b is None else b.reshape(-1, 1)
        self.y = np.zeros(n_samples) if y is None else y.reshape(-1, 1)
        self.idx_order = range(self.A.shape[0]) if idx_order is None else idx_order

        # Verify arguments?

        # Initialize chain to initial mean
        self.epochs = []
        self.solution_chains = []
        self.errors = []

    def solve(self, num_epochs=1, method="mud"):
        """
        Iterative Solutions
        Performs num_epochs iterations of estimates

        """
        m_init = (
            self.mu_i
            if len(self.solution_chains) == 0
            else self.solution_chains[-1][-1]
        )
        solutions = [m_init]
        for _ in range(0, num_epochs):
            epoch = []
            solutions = [solutions[-1]]
            for i in self.idx_order:
                # Add next sub-problem to chain
                epoch.append(
                    LinearGaussianProblem(
                        self.A[i, :],
                        self.b[i],
                        self.y[i],
                        mean=solutions[-1],
                        cov=self.cov,
                        data_cov=self.data_cov,
                    )
                )

                # Solve next mud problem
                solutions.append(epoch[-1].solve(method=method))

            self.epochs.append(epoch)
            self.solution_chains.append(solutions)

        return self.solution_chains[-1][-1]

    def get_errors(self, ref_param):
        """
        Get errors with resepct to a reference parameter

        """
        solutions = np.concatenate([x[1:] for x in self.solution_chains])
        if len(solutions) != len(self.errors):
            self.errors = [np.linalg.norm(s - ref_param) for s in solutions]
        return self.errors

    def plot_chain(self, ref_param, ax=None, color="k", s=100, **kwargs):
        """
        Plot chain of solutions and contours
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for e, chain in enumerate(self.solution_chains):
            num_steps = len(chain)
            current_point = chain[0]
            ax.scatter(current_point[0], current_point[1], c="b", s=s)
            for i in range(0, num_steps):
                next_point = chain[i]
                points = np.hstack([current_point, next_point])
                ax.plot(points[0, :], points[1, :], c=color)
                current_point = next_point
            ax.scatter(current_point[0], current_point[1], c="g", s=s)
            ax.scatter(ref_param[0], ref_param[1], c="r", s=s)
        self.plot_contours(
            ref_param, ax=ax, subset=self.idx_order, color=color, s=s, **kwargs
        )

    def plot_chain_error(
        self, ref_param, ax=None, alpha=1.0, color="k", label=None, s=100, fontsize=12
    ):
        """
        Plot error over iterations
        """
        _ = self.get_errors(ref_param)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.set_yscale("log")
        ax.plot(self.errors, color=color, alpha=alpha, label=label)
        ax.set_ylabel("$||\\lambda - \\lambda^\\dagger||$", fontsize=fontsize)
        ax.set_xlabel("Iteration step", fontsize=fontsize)
