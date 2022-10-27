import pickle
from typing import Callable, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt  # type: ignore
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats import rv_continuous  # type: ignore
from scipy.stats.contingency import margins  # type: ignore

from mud.plot import plot_dist, plot_vert_line
from mud.preprocessing import pca, svd
from mud.util import add_noise, fit_domain, make_2d_unit_mesh, null_space, set_shape

try:
    import xarray as xr  # type: ignore

    xr_avial = True
except ModuleNotFoundError:
    xr_avail = False
    pass


class DensityProblem(object):
    """
    Sets up Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model parameters
    using observed data and predicted data from the model.

    Attributes
    ----------
    X : ArrayLike
        Array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
        If 1 dimensional input is passed, assumed that it represents repeated
        samples of a 1-dimensional parameter.
    y : ArrayLike
        Array containing push-forward values of parameters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : ArrayLike
        Array containing ranges of each parameter value in the parameter
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

    >>> from mud.examples.simple import identity_1D_density_prob as I1D

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

    Expectation value of r, ratio of observed and predicted distribution, should
    be near 1 if predictability assumption is satisfied.

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
        domain: ArrayLike = None,
        weights: ArrayLike = None,
        normalize: bool = False,
        pad_domain: float = 0.1,
    ):

        self.X = set_shape(np.array(X), (1, -1))
        self.y = set_shape(np.array(y), (-1, 1))

        # These will be updated in set_ and fit() functions
        self._r = None  # Ratio of observed to predicted
        self._up = None  # Updated values
        self._in = None  # Initial values
        self._pr = None  # Predicted values
        self._ob = None  # Observed values
        self._in_dist = None  # Initial distribution
        self._pr_dist = None  # Predicted distribution
        self._ob_dist = None  # Observed distribution

        if domain is not None:
            # Assert domain passed in is consistent with data array
            self.domain = set_shape(np.array(domain), (1, -1))
            assert (
                self.domain.shape[0] == self.n_params
            ), f"Size mismatch: domain: {self.domain.shape}, params: {self.X.shape}"

        else:
            self.domain = fit_domain(self.X)

        # Initialize weights
        self.set_weights(weights, normalize=normalize)

    @property
    def n_params(self):
        return self.X.shape[1]

    @property
    def n_features(self):
        return self.y.shape[1]

    @property
    def n_samples(self):
        return self.y.shape[0]

    def set_weights(self, weights: ArrayLike = None, normalize: bool = False):
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
        normalize : bool, default=False
            Whether to normalize the weights vector.

        Returns
        -------

        Warnings
        --------
        Resetting weights will delete the predicted and updated distribution
        values in the class, requiring a re-run of adequate `set_` methods
        and/or `fit()` to reproduce with new weights.
        """
        if weights is None:
            w = np.ones(self.X.shape[0])
        else:
            w = np.array(weights)

            # Reshape to 2D
            w = w.reshape(1, -1) if w.ndim == 1 else w

            # assert appropriate size
            assert self.n_samples == w.shape[1], f"`weights` must size {self.n_samples}"

            # Multiply weights column wise for stacked weights
            w = np.prod(w, axis=0)

            if normalize:
                w = np.divide(w, np.linalg.norm(w))

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

    def set_initial(self, distribution: Optional[rv_continuous] = None):
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
        Setting initial distribution resets the predicted and updated
        distributions, so make sure to set the initial first.
        """
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()

        self._in = distribution.pdf(self.X).prod(axis=1)
        self._in_dist = distribution
        self._up = None
        self._pr = None
        self._pr_dist = None

    def set_predicted(
        self,
        distribution: rv_continuous = None,
        bw_method: Union[str, Callable, np.generic] = None,
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
        bw_method : str, scalar, or `Callable`, optional
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
            If specified, any extra keyword arguments will be passed along to
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

        Constructs the updated distribution by fitting observed data to
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
            \\lambda^{MUD} := \\mathrm{argmax} \\pi_{up}(\\lambda)
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

        Returns the best estimate for most likely parameter values for the
        given model data using the data-consistent framework.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
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

    # TODO: update documentation
    def plot_param_space(
        self,
        param_idx: int = 0,
        true_val: ArrayLike = None,
        ax: plt.Axes = None,
        x_range: Union[list, np.ndarray] = None,
        ylim: float = None,
        pad_ratio: float = 0.05,
        aff: int = 100,
        in_opts={"color": "b", "linestyle": "-", "label": r"$\pi_\mathrm{init}$"},
        up_opts={"color": "k", "linestyle": "-.", "label": r"$\pi_\mathrm{update}$"},
        win_opts=None,
        mud_opts={"color": "g", "label": r"$\lambda^\mathrm{MUD}$"},
        true_opts={"color": "r", "label": r"$\lambda^{\dagger}$"},
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
            'label':'Initial'}``. To suppress plotting, pass in ``None``
            explicitly.
        up_opts : dict, optional
            Plotting option for updated distribution line. Defaults to
            ``{'color':'k', 'linestyle':'-.','linewidth':4,
            'label':'Updated'}``. To suppress plotting, pass in ``None``
            explicitly.
        win_opts : dict, optional
            Plotting option for weighted initial distribution line. Defaults to
            ``{'color':'g', 'linestyle':'--','linewidth':4,
            'label':'Weighted Initial'}``. To suppress plotting, pass in
            ``None`` explicitly.

        Returns
        -------
        """
        # Default options for plotting figures
        io = {"color": "b", "linestyle": "--", "label": r"$\pi_\\mathrm{init}$"}
        uo = {"color": "k", "linestyle": "-.", "label": r"$\pi_\\mathrm{update}$"}
        wo = {"color": "b", "linestyle": ":", "label": r"$\\tilde{\pi}_\\mathrm{init}$"}
        mo = {"color": "g", "label": r"$\lambda^\mathrm{MUD}$"}
        to = {"color": "r", "linestyle": "-.", "label": r"$\lambda^{\dagger}$"}

        # Create plot if one isn't passed in
        _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

        # Default x_range to full domain of all parameters
        if x_range is None:
            x_range = fit_domain(min_max_bounds=self.domain, pad_ratio=pad_ratio)

        # Plot distributions for all not set to None
        assert self._in_dist is not None
        if in_opts is not None:
            io.update(in_opts)
            plot_dist(
                self._in_dist,
                x_range,
                idx=param_idx,
                ax=ax,
                source="pdf",
                aff=aff,
                **io,
            )
        if up_opts is not None:
            uo.update(up_opts)
            if self._r is not None:
                up_kde = gkde(self.X.T, weights=self._r * self._weights)
            else:
                up_kde = gkde(self.X.T, weights=None)
            plot_dist(
                up_kde, x_range, idx=param_idx, ax=ax, source="kde", aff=aff, **uo
            )
        if win_opts is not None:
            wo.update(win_opts)
            w_kde = gkde(self.X.T, weights=self._weights)
            plot_dist(w_kde, x_range, idx=param_idx, ax=ax, source="kde", aff=aff, **wo)
        if mud_opts is not None:
            mo.update(mud_opts)
            mud_pt = self.estimate()
            plot_vert_line(ax, mud_pt[param_idx], ylim=ylim, **mo)
        if true_val is not None and true_opts:
            true_val = np.array(true_val)
            to.update(true_opts)
            plot_vert_line(ax, true_val[param_idx], ylim=ylim, **to)

        ax.set_xlabel(rf"$\lambda_{param_idx+1}$")

        return ax

    def plot_obs_space(
        self,
        obs_idx: int = 0,
        ax: plt.Axes = None,
        y_range: ArrayLike = None,
        aff=100,
        ob_opts={"color": "r", "linestyle": "-", "label": r"$\pi_\mathrm{obs}$"},
        pr_opts={"color": "b", "linestyle": "-", "label": r"$\pi_\mathrm{pred}$"},
        pf_opts={"color": "k", "linestyle": "-.", "label": r"$\pi_\mathrm{pf-pr}$"},
    ):
        """
        Plot probability distributions over parameter space

        Observed distribution is plotted using the distribution function passed
        to :meth:`set_observed` (or default). The predicted distribution is
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
            'label':'Observed'}``. To suppress plotting, pass in ``None``.
        pr_opts : dict, optional
            Plotting option for predicted distribution line. Defaults to
            ``{'color':'b', 'linestyle':'--','linewidth':4,
            'label':'PF of Initial'}``. To suppress plotting, pass in ``None``.
        pf_opts : dict, optional
            Plotting option for push-forward of updated distribution line.
            Defaults to ``{'color':'k', 'linestyle':'-.','linewidth':4,
            'label':'PF of Updated'}``. To suppress plotting, pass in
            ``None``.

        Returns
        -------
        """
        # observed, predicted, and push-forward opts respectively
        oo = {"color": "r", "linestyle": "-", "label": r"$\pi_\mathrm{obs}$"}
        po = {"color": "b", "linestyle": "-", "label": r"$\pi_\mathrm{pred}$"}
        fo = {"color": "k", "linestyle": "-.", "label": r"$\pi_\mathrm{pf-pr}$"}

        # Create plot if one isn't passed in
        _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

        # Default range is (-1,1) over given observable index
        # TODO: Infer range from predicted y vals
        y_range = fit_domain(self.y) if y_range is None else np.array(y_range)
        if y_range.shape[0] != self.n_features:
            raise ValueError("Invalid domain dimension")

        assert self._ob_dist is not None, "Observed dist empty"
        if ob_opts:
            oo.update(ob_opts)
            plot_dist(
                self._ob_dist, y_range, idx=obs_idx, ax=ax, source="pdf", aff=aff, **oo
            )
        if pr_opts:
            po.update(pr_opts)
            source = "pdf" if isinstance(self._pr_dist, type(dist.uniform())) else "kde"
            plot_dist(
                self._pr_dist, y_range, idx=obs_idx, ax=ax, source=source, aff=aff, **po
            )
        if pf_opts:
            fo.update(pf_opts)

            # Compute PF of updated
            if self._r is not None:
                pf_kde = gkde(self.y.T, weights=self._weights * self._r)
            else:
                pf_kde = gkde(self.y.T, weights=None)
            plot_dist(pf_kde, y_range, idx=obs_idx, ax=ax, source="kde", aff=aff, **fo)

        return ax

    def plot_qoi(self, idx_x: int = 0, idx_y: int = 1, ax: plt.Axes = None, **kwargs):
        """
        Plot 2D plot over two indices of y space.

        Parameters
        ----------
        idx_x: int, default=0
            Index of observable value (column of y) to use as x value in plot.
        idx_y: int, default=1
            Index of observable value (column of y) to use as y value in plot.
        ax : :class:`matplotlib.axes.Axes`, optional
            Axes to plot distributions on. If non specified, a figure will
            be initialized to plot on.
        kwargs : dict, optional
            Additional keyword arguments will be passed to
            `matplotlib.pyplot.scatter`.

        Returns
        -------
        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.y[:, idx_x], self.y[:, idx_y], **kwargs)

        return ax

    def plot_params_2d(
        self,
        x_1: int = 0,
        x_2: int = 1,
        y: int = 0,
        contours: bool = False,
        colorbar: bool = True,
        ax: plt.Axes = None,
        label=True,
        **kwargs,
    ):
        """
        2D plot over two indices of param space, optionally contoured by a y
        value.

        Parameters
        ----------
        x_1 : int, default=0
            Index of param value (column of X) to use as x value in plot.
        x_2 : int, default=1
            Index of param value (column of X) to use as y value in plot.
        y: int, optional
            Index of observable (column of y) to use as z value in contour plot.
        ax : :class:`matplotlib.axes.Axes`, optional
            Axes to plot distributions on. If non specified, a figure will
            be initialized to plot on.
        kwargs : dict, optional
            Additional keyword arguments will be passed to
            `matplotlib.pyplot.scatter`.

        Returns
        -------
        """
        if self.n_params < 2:
            raise AttributeError("2D plot requires at least 2 dim param space.")
        if x_1 >= self.n_params or x_2 >= self.n_params:
            raise ValueError("indices {x_1+1},{x_2+1} > dim {self.n_params}")

        c = None
        if y >= self.n_features:
            raise ValueError("index {y+1} > dim {self.n_params}")
        c = self.y[:, y]

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

        sp = ax.scatter(self.X[:, 0], self.X[:, 1], c=c)

        if contours:
            ax.tricontour(self.X[:, 0], self.X[:, 1], c, levels=20)
            ax.set_title(f"$q_{y+1}$")

        if colorbar and c is not None:
            fig = plt.gcf()
            fig.colorbar(sp)

        ax.set_xlabel(rf"$\lambda_{x_1+1}$")
        ax.set_ylabel(rf"$\lambda_{x_2+1}$")

        return ax


class BayesProblem(object):
    """
    Sets up Bayesian Inverse Problem for parameter identification

    Parameters
    ----------
    X : ndarray
        2D array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
    y : ndarray
        array containing push-forward values of parameters samples through the
        forward model. These samples will form the data-likelihood
        distribution.
    domain : array_like, optional
        2D Array containing ranges of each parameter value in the parameter
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
        # Set and validate inputs. Note we reshape inputs as necessary
        def set_shape(x, y):
            return x.reshape(y) if x.ndim < 2 else x

        self.X = set_shape(np.array(X), (1, -1))
        self.y = set_shape(np.array(y), (1, -1))

        if domain is not None:
            # Assert domain passed in is consistent with data array
            self.domain = set_shape(np.array(domain), (1, -1))
            assert self.domain.shape[0] == self.n_params
        else:
            self.domain = fit_domain(self.X)

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

    # TODO: Make changes similar to DensityProblem class
    def plot_param_space(
        self,
        param_idx=0,
        ax=None,
        x_range=None,
        aff=1000,
        pr_opts={"color": "b", "linestyle": "--", "linewidth": 4, "label": "Prior"},
        ps_opts={"color": "g", "linestyle": ":", "linewidth": 4, "label": "Posterior"},
        map_opts={"color": "g", "label": r"$\lambda^\mathrm{MAP}$"},
        true_opts={"color": "r", "linestyle": "-.", "label": r"$\lambda^{\dagger}$"},
        true_val=None,
    ):
        """
        Plot probability distributions over parameter space

        """
        # Default options for plotting figures
        pro = {"color": "b", "linestyle": "--", "linewidth": 4, "label": "Prior"}
        pso = {"color": "g", "linestyle": ":", "linewidth": 4, "label": "Posterior"}
        mo = {"color": "g", "label": r"$\lambda^\mathrm{MAP}$"}
        to = {"color": "r", "linestyle": "-.", "label": r"$\lambda^{\dagger}$"}

        # Create plot if one isn't passed in
        _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

        # Default x_range to full domain of all parameters
        x_range = x_range if x_range is not None else self.domain

        # Plot distributions for all not set to None
        if pr_opts is not None:
            pro.update(pr_opts)
            plot_dist(
                self._pr_dist,
                x_range,
                idx=param_idx,
                ax=ax,
                source="pdf",
                aff=aff,
                **pro,
            )
        if ps_opts is not None:
            pso.update(ps_opts)
            self.estimate()
            ps_kde = gkde(self.X.T, weights=self._ps)
            plot_dist(
                ps_kde, x_range, idx=param_idx, ax=ax, source="kde", aff=aff, **pso
            )
        if map_opts is not None:
            mo.update(map_opts)
            map_pt = self.estimate()
            plot_vert_line(ax, map_pt[param_idx], **mo)
        if true_val is not None and true_opts:
            to.update(true_opts)
            plot_vert_line(ax, true_val[param_idx], **to)

        ax.set_xlabel(rf"$\lambda_{param_idx+1}$")

        return ax

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
        lo = {
            "color": "r",
            "linestyle": "-",
            "linewidth": 4,
            "label": "Data-Likelihood",
        }
        po = {
            "color": "g",
            "linestyle": ":",
            "linewidth": 4,
            "label": "PF of Posterior",
        }
        # Create plot if one isn't passed in

        _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

        # Default range is (-1,1) over each observable variable
        if y_range is None:
            y_range = np.repeat([[-1, 1]], self.y.shape[1], axis=0)

        # Build grid of points over range to compute marginals
        XXX = np.meshgrid(*[np.linspace(i, j, aff)[:-1] for i, j in y_range])
        grid_points = np.vstack([x.ravel() for x in XXX])
        y_plot = np.linspace(y_range[obs_idx, 0], y_range[obs_idx, 1], aff)[: aff - 1]

        if ll_opts is not None:
            lo.update(ll_opts)

            if self._ll is None:
                raise ValueError("Likelihood not set. Run fit()")

            # Compute observed distribution using stored pdf
            ll_plot = margins(
                np.reshape(self._ll_dist.pdf(grid_points).T.prod(axis=1), XXX[0].shape)
            )[obs_idx].reshape(-1)

            # Plot pf of initial
            ax.plot(y_plot, ll_plot, **lo)

        if pf_opts is not None:
            po.update(pf_opts)

            # Compute PF of posterior
            pf_kde = gkde(self.y.T, weights=self._ps)
            pf_p = margins(np.reshape(pf_kde(grid_points).T, XXX[0].shape))[
                obs_idx
            ].reshape(-1)

            # Plut pf of updated
            ax.plot(y_plot, pf_p, **po)

        return ax


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
        2D Array defining linear transformation from model parameter space to
        model output space.
    y : ArrayLike
        1D Array containing observed values of Q(\\lambda)
        Array containing push-forward values of parameters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : ArrayLike
        Array containing ranges of each parameter value in the parameter
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
        A = np.array(A)
        self.A = A if A.ndim == 2 else A.reshape(1, -1)
        ns, di = self.A.shape

        # Initialize to defaults - Reshape everything into 2D arrays.
        self.b = np.zeros((ns, 1)) if b is None else np.array(b).reshape(-1, 1)
        self.y = np.zeros((ns, 1)) if y is None else np.array(y).reshape(-1, 1)
        self.mean_i = (
            np.zeros((di, 1)) if mean_i is None else np.array(mean_i).reshape(-1, 1)
        )
        self.cov_i = np.eye(di) if cov_i is None else np.array(cov_i)
        self.cov_o = np.eye(ns) if cov_o is None else np.array(cov_o)

        # How much to scale regularization terms
        self.alpha = alpha if alpha is not None else 1.0

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

        # Data-Consistent Term - "un-regularization" in data-informed directions
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

        if method == "mud_alt" or method == "all":
            up_cov = self.updated_cov(A=_A, init_cov=a_cov_i, data_cov=_cov_o)
            update = up_cov @ _A.T @ np.linalg.inv(_cov_o)
            self.mud_alt = self.mean_i + update @ z
            self.up_cov = up_cov

        if method == "map" or method == "all":
            co_inv = np.linalg.inv(_cov_o)
            cov_p = np.linalg.inv(_A.T @ co_inv @ _A + np.linalg.inv(a_cov_i))
            update = cov_p @ _A.T @ co_inv
            self.map = self.mean_i + update @ z
            self.cov_p = cov_p

        if method == "ls" or method == "all":
            # Compute ls solution from pinv method
            self.ls = np.linalg.pinv(_A) @ mean_o

        # Return solution or all solutions
        if method == "all":
            return (self.mud, self.map, self.ls)
            # return (self.mud, self.mud_alt, self.map, self.ls)
        else:
            return self.__getattribute__(method)

    def updated_cov(self, A=None, init_cov=None, data_cov=None):
        """
        We start with the posterior covariance from ridge regression
        Our matrix R = init_cov^(-1) - X.T @ pred_cov^(-1) @ X
        replaces the init_cov from the posterior covariance equation.
        Simplifying, this is given as the following, which is not used
        due to issues of numerical stability (a lot of inverse operations).

        up_cov = (X.T @ np.linalg.inv(data_cov) @ X + R )^(-1)
        up_cov = np.linalg.inv(\
            X.T@(np.linalg.inv(data_cov) - inv_pred_cov)@X + \
            np.linalg.inv(init_cov) )

        We return the updated covariance using a form of it derived
        which applies Hua's identity in order to use Woodbury's identity.

        Check using alternate for updated_covariance.

        >>> from mud.base import LinearGaussianProblem as LGP
        >>> lg2 = LGP(A=np.eye(2))
        >>> lg2.updated_cov()
        array([[1., 0.],
               [0., 1.]])
        >>> lg3 = LGP(A=np.eye(2)*2)
        >>> lg3.updated_cov()
        array([[0.25, 0.  ],
               [0.  , 0.25]])
        >>> lg3 = LGP(A=np.eye(3)[:, :2]*2)
        >>> lg3.updated_cov()
        array([[0.25, 0.  ],
               [0.  , 0.25]])
        >>> lg3 = LGP(A=np.eye(3)[:, :2]*2, cov_i=np.eye(2))
        >>> lg3.updated_cov()
        array([[0.25, 0.  ],
               [0.  , 0.25]])
        """
        X = A if A is not None else self.A
        if init_cov is None:
            init_cov = self.cov_i
        else:
            assert X.shape[1] == init_cov.shape[1]

        if data_cov is None:
            data_cov = self.cov_o
        else:
            assert X.shape[0] == data_cov.shape[1]

        pred_cov = X @ init_cov @ X.T
        inv_pred_cov = np.linalg.pinv(pred_cov)
        # pinv b/c inv unstable for rank-deficient A

        # Form derived via Hua's identity + Woodbury
        K = init_cov @ X.T @ inv_pred_cov
        up_cov = init_cov - K @ (pred_cov - data_cov) @ K.T

        return up_cov

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

        # Plot line connecting initial value and solution
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
        Plot contour map of functionals being minimized over input space
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


class LinearWMEProblem(LinearGaussianProblem):
    """Linear Inverse Problems using the Weighted Mean Error Map"""

    def __init__(
        self,
        operators,
        data,
        sigma,
        y=None,
        mean_i=None,
        cov_i=None,
        cov_o=None,
        alpha=1.0,
    ):

        if isinstance(sigma, (float, int)):
            sigma = [sigma] * len(data)

        results = [
            self._transform_linear_map(o, d, s)
            for o, d, s in zip(operators, data, sigma)
        ]
        operators = [r[0] for r in results]
        datas = [r[1] for r in results]
        A, B = np.vstack(operators), np.vstack(datas)

        super().__init__(
            A=A, b=B, y=y, mean_i=mean_i, cov_i=cov_i, cov_o=cov_o, alpha=alpha
        )

    def _transform_linear_map(self, operator, data, std):
        """
        Takes a linear map `operator` of size (len(data), dim_input)
        or (1, dim_input) for repeated observations, along with
        a vector `data` representing observations. It is assumed
        that `data` is formed with `M@truth + sigma` where `sigma ~ N(0, std)`

        This then transforms it to the MWE form expected by the DCI framework.
        It returns a matrix `A` of shape (1, dim_input) and np.float `b`
        and transforms it to the MWE form expected by the DCI framework.

        >>> from mud.base import LinearWMEProblem as LWP
        >>> operators = [np.ones((10, 2))]
        >>> x = np.array([0.5, 0.5]).reshape(-1, 1)
        >>> sigma = 1
        >>> data = [X @ x for X in operators]
        >>> lin_wme_prob = LWP(operators, data, sigma)
        >>> lin_wme_prob.y
        array([[0.]])
        >>> lin_wme_prob = LWP(operators, data, [sigma]*10)
        >>> lin_wme_prob.y
        array([[0.]])
        >>> operators = [np.array([[1, 1]])]
        >>> lin_wme_prob = LWP(operators, data, sigma)
        >>> lin_wme_prob.y
        array([[0.]])
        """
        if isinstance(data, np.ndarray):
            data = data.ravel()

        num_observations = len(data)

        if operator.shape[0] > 1:  # if not repeated observations
            assert (
                operator.shape[0] == num_observations
            ), f"Operator shape mismatch, op={operator.shape}, obs={num_observations}"
            if isinstance(std, (float, int)):
                std = np.array([std] * num_observations)
            if isinstance(std, (list, tuple)):
                std = np.array(std)
            assert len(std) == num_observations, "Standard deviation shape mismatch"
            assert 0 not in np.round(std, 14), "Std must be > 1E-14"
            D = np.diag(1.0 / (std * np.sqrt(num_observations)))
            A = np.sum(D @ operator, axis=0)
        else:
            if isinstance(std, (list, tuple, np.ndarray)):
                raise ValueError("For repeated measurements, pass a float for std")
            assert std > 1e-14, "Std must be > 1E-14"
            A = np.sqrt(num_observations) / std * operator

        b = -1.0 / np.sqrt(num_observations) * np.sum(np.divide(data, std))
        return A, b


class IterativeLinearProblem(LinearGaussianProblem):
    def __init__(
        self, A, b, y=None, mu_i=None, cov=None, data_cov=None, idx_order=None
    ):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)

        # Initialize to defaults - Reshape everything into 2D arrays.
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
        Get errors with respect to a reference parameter

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


class SpatioTemporalProblem(object):
    """
    Class for parameter estimation problems related to spatio-temporal problems.
    equation models of real world systems. Uses a QoI map of weighted
    residuals between simulated data and measurements to do inversion

    Attributes
    ----------
    TODO: Finish

    Methods
    -------
    TODO: Finish


    """

    def __init__(self, df=None):

        self._domain = None
        self._lam = None
        self._data = None
        self._measurements = None
        self._true_lam = None
        self._true_vals = None
        self._sample_dist = None
        self.sensors = None
        self.times = None
        self.qoi = None
        self.pca = None
        self.std_dev = None

        if df is not None:
            self.load(df)

    @property
    def n_samples(self) -> int:
        if self.lam is None:
            raise AttributeError("lambda not yet set.")
        return self.lam.shape[0]

    @property
    def n_qoi(self) -> int:
        if self.qoi is None:
            raise AttributeError("qoi not yet set.")
        return self.qoi.shape[1]

    @property
    def n_sensors(self) -> int:
        if self.sensors is None:
            raise AttributeError("sensors not yet set.")
        return self.sensors.shape[0]

    @property
    def n_ts(self) -> int:
        if self.times is None:
            raise AttributeError("times not yet set.")
        return self.times.shape[0]

    @property
    def n_params(self) -> int:
        return self.domain.shape[0]

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam):
        lam = np.array(lam)
        lam = lam.reshape(-1, 1) if lam.ndim == 1 else lam

        if self.domain is not None:
            if lam.shape[1] != self.n_params:
                raise ValueError("Parameter dimensions do not match domain specified.")
        else:
            # TODO: Determine domain from min max in parameters
            self.domain = np.vstack([lam.min(axis=0), lam.max(axis=0)]).T
        if self.sample_dist is None:
            # Assume uniform distribution by default
            self.sample_dist = "u"

        self._lam = lam

    @property
    def lam_ref(self):
        return self._lam_ref

    @lam_ref.setter
    def lam_ref(self, lam_ref):
        if self.domain is None:
            raise AttributeError("domain not yet set.")
        lam_ref = np.reshape(lam_ref, (-1))
        for idx, lam in enumerate(lam_ref):
            if (lam < self.domain[idx][0]) or (lam > self.domain[idx][1]):
                raise ValueError(
                    f"lam_ref at idx {idx} must be inside {self.domain[idx]}."
                )
        self._lam_ref = lam_ref

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        domain = np.reshape(domain, (-1, 2))
        if self.lam is not None:
            if self.domain.shape[0] != self.lam.shape[1]:
                raise ValueError("Domain and parameter array dimension mismatch.")
            min_max = np.vstack([self.lam.min(axis=0), self.lam.max(axis=0)]).T
            if not all(
                [all(domain[:, 0] <= min_max[:, 0]), all(domain[:, 1] >= min_max[:, 1])]
            ):
                raise ValueError("Parameter values exist outside of specified domain")

        self._domain = domain

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        dim = data.shape
        ndim = data.ndim
        if ndim == 1:
            data = np.reshape(data, (-1, 1))
        if ndim == 3:
            # Expected to be in (# samples x # sensors # # timesteps)
            data = np.reshape(data, (dim[0], -1))

        dim = data.shape
        ndim = data.ndim
        if self.sensors is None and self.times is None:
            self.sensors = np.array([0])
            self.times = np.arange(0, dim[1])
        if self.sensors is not None and self.times is None:
            if self.sensors.shape[0] != dim[1]:
                raise ValueError(
                    "Dimensions of simulated data does not match number of sensors"
                )
            self.times = np.array([0])
        if self.sensors is None and self.times is not None:
            if self.times.shape[0] != dim[1]:
                raise ValueError(
                    "Dimensions of simulated data does not match number of timesteps"
                )
            self.sensors = np.array([0])
        if self.sensors is not None and self.times is not None:
            # Assume data is already flattened, check dimensions match
            if self.times.shape[0] * self.sensors.shape[0] != dim[1]:
                raise ValueError(
                    "Dimensions of simulated data != (timesteps x sensors)"
                )

        # Flatten data_data into 2d array
        self._data = data

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, measurements):
        measurements = np.reshape(measurements, (self.n_sensors * self.n_ts, 1))
        self._measurements = measurements

    @property
    def true_vals(self):
        return self._true_vals

    @true_vals.setter
    def true_vals(self, true_vals):
        true_vals = np.reshape(true_vals, (self.n_sensors * self.n_ts, 1))
        self._true_vals = true_vals

    @property
    def sample_dist(self):
        return self._sample_dist

    @sample_dist.setter
    def sample_dist(self, dist):
        if dist not in ["u", "n"]:
            raise ValueError(
                "distribution could not be inferred. Must be from ('u', 'n')"
            )
        self._sample_dist = dist

    def get_closest_to_measurements(
        self,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
    ):
        """
        Get closest simulated data point to measured data in $l^2$-norm.
        """
        lam, times, sensors, sub_data, sub_meas = self.sample_data(
            samples_mask=samples_mask,
            times_mask=times_mask,
            sensors_mask=sensors_mask,
        )
        closest_idx = np.argmin(np.linalg.norm(sub_data - sub_meas.T, axis=1))
        closest_lam = lam[closest_idx, :].ravel()

        return closest_lam

    def get_closest_to_true_vals(
        self,
    ):
        """
        Get closest simulated data point to noiseless true values in $l^2$-norm.

        Note for now no sub-sampling implemented here.
        """
        if self.true_vals is None:
            raise AttributeError("True values is not set")
        closest_idx = np.argmin(np.linalg.norm(self.data - self.true_vals.T, axis=1))
        closest_lam = self.lam[closest_idx, :].ravel()

        return closest_lam

    def measurements_from_reference(self, ref=None, std_dev=None, seed=None):
        """
        Add noise to a reference solution.
        """
        if ref is not None:
            self._true_vals = ref
        if std_dev is not None:
            self.std_dev = std_dev
        if self.true_vals is None or self.std_dev is None:
            raise AttributeError(
                "Must set reference solution and std_dev first or pass as arguments."
            )
        self.measurements = np.reshape(
            add_noise(self.true_vals.ravel(), self.std_dev, seed=seed),
            self.true_vals.shape,
        )

    def load(
        self,
        df,
        lam="lam",
        data="data",
        **kwargs,
    ):
        """
        Load data from a file on disk for a PDE parameter estimation problem.

        Parameters
        ----------
        fname : str
            Name of file on disk. If ends in '.nc' then assumed to be netcdf
            file and the xarray library is used to load it. Otherwise the
            data is assumed to be pickled data.

        Returns
        -------
        ds : dict,
            Dictionary containing data from file for PDE problem class

        """
        if type(df) == str:
            try:
                if df.endswith("nc") and xr_avail:
                    ds = xr.load_dataset(df)
                else:
                    with open(df, "rb") as fp:
                        ds = pickle.load(fp)
            except FileNotFoundError:
                raise FileNotFoundError(f"Couldn't find data file {df}")
        else:
            ds = df

        def get_set_val(f, v):
            if f in ds.keys():
                self.__setattr__(f, ds[v])
            elif v is not None and type(v) != str:
                self.__setattr__(f, v)

        field_names = {
            "sample_dist": "sample_dist",
            "domain": "domain",
            "sensors": "sensors",
            "times": "times",
            "lam_ref": "lam_ref",
            "std_dev": "std_dev",
            "true_vals": "true_vals",
            "measurements": "measurements",
        }
        field_names.update(kwargs)
        for f, v in field_names.items():
            get_set_val(f, v)

        get_set_val("lam", lam)
        get_set_val("data", data)

        return ds

    def validate(
        self,
        check_meas=True,
        check_true=False,
    ):
        """Validates if class has been set-up appropriately for inversion"""
        req_attrs = ["domain", "lam", "data"]
        if check_meas:
            req_attrs.append("measurements")
        if check_true:
            req_attrs.append("true_lam")
            req_attrs.append("true_vals")

        missing = [x for x in req_attrs if self.__getattribute__(x) is None]
        if len(missing) > 0:
            raise ValueError(f"Missing attributes {missing}")

    def sample_data(
        self,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
    ):
        if self.data is None:
            raise AttributeError("data not set yet.")

        sub_data = np.reshape(self.data, (self.n_samples, self.n_sensors, self.n_ts))
        if self.measurements is not None:
            sub_meas = np.reshape(self.measurements, (self.n_sensors, self.n_ts))
        else:
            sub_meas = None

        sub_times = self.times
        sub_sensors = self.sensors
        sub_lam = self.lam
        if samples_mask is not None:
            sub_lam = self.lam[samples_mask, :]
            sub_data = sub_data[samples_mask, :, :]
        if times_mask is not None:
            sub_times = np.reshape(self.times[times_mask], (-1, 1))
            sub_data = sub_data[:, :, times_mask]
            if sub_meas is not None:
                sub_meas = sub_meas[:, times_mask]
        if sensors_mask is not None:
            sub_sensors = np.reshape(self.sensors[sensors_mask], (-1, 2))
            sub_data = sub_data[:, sensors_mask, :]
            if sub_meas is not None:
                sub_meas = sub_meas[sensors_mask, :]

        sub_data = np.reshape(sub_data, (-1, sub_times.shape[0] * sub_sensors.shape[0]))
        if sub_meas is not None:
            sub_meas = np.reshape(sub_meas, (len(sub_times) * len(sub_sensors)))

        return sub_lam, sub_times, sub_sensors, sub_data, sub_meas

    def sensor_contour_plot(
        self, idx=0, c_vals=None, ax=None, mask=None, fill=True, colorbar=True, **kwargs
    ):
        """
        Plot locations of sensors in space
        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

        sensors = self.sensors[mask, :] if mask is not None else self.sensors
        contours = c_vals if c_vals is not None else self.data[idx, :].ravel()

        if fill:
            tc = ax.tricontourf(sensors[:, 0], sensors[:, 1], contours, **kwargs)
        else:
            tc = ax.tricontour(sensors[:, 0], sensors[:, 1], contours, **kwargs)

        if colorbar:
            fig = plt.gcf()
            fig.colorbar(tc)

        return ax

    def sensor_scatter_plot(self, ax=None, mask=None, colorbar=None, **kwargs):
        """
        Plot locations of sensors in space
        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

        sensors = self.sensors[mask, :] if mask is not None else self.sensors

        sp = plt.scatter(sensors[:, 0], sensors[:, 1], **kwargs)

        if colorbar and "c" in kwargs.keys():
            fig = plt.gcf()
            fig.colorbar(sp)

        return ax

    def plot_ts(
        self,
        ax=None,
        samples=None,
        times=None,
        sensor_idx=0,
        max_plot=100,
        meas_kwargs={},
        samples_kwargs={},
        alpha=0.1,
    ):
        """
        Plot time series data
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 1, 1)

        lam, times, _, sub_data, sub_meas = self.sample_data(
            samples_mask=samples, times_mask=times, sensors_mask=sensor_idx
        )
        num_samples = sub_data.shape[0]
        max_plot = num_samples if max_plot > num_samples else max_plot

        # Plot measured time series
        def_kwargs = {
            "color": "k",
            "marker": "^",
            "label": "$\\zeta_{obs}$",
            "zorder": 50,
            "s": 2,
        }
        def_kwargs.update(meas_kwargs)
        _ = plt.scatter(times, sub_meas, **def_kwargs)

        # Plot simulated data time series
        def_sample_kwargs = {"color": "r", "linestyle": "-", "zorder": 1, "alpha": 0.1}
        def_sample_kwargs.update(samples_kwargs)
        for i, idx in enumerate(np.random.choice(num_samples, max_plot)):
            if i != (max_plot - 1):
                _ = ax.plot(times, sub_data[i, :], **def_sample_kwargs)
            else:
                _ = ax.plot(
                    times,
                    sub_data[i, :],
                    "r-",
                    alpha=alpha,
                    label=f"Sensor {sensor_idx}",
                )

        return ax

    def mud_problem(
        self,
        method="pca",
        data_weights=None,
        sample_weights=None,
        num_components=2,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
    ):
        """Build QoI Map Using Data and Measurements"""

        # TODO: Finish sample data implementation
        lam, times, sensors, sub_data, sub_meas = self.sample_data(
            samples_mask=samples_mask,
            times_mask=times_mask,
            sensors_mask=sensors_mask,
        )
        residuals = (sub_meas - sub_data) / self.std_dev
        sub_n_samples = sub_data.shape[0]

        if data_weights is not None:
            data_weights = np.reshape(data_weights, (-1, 1))
            if data_weights.shape[0] != self.n_sensors * self.n_ts:
                raise ValueError(
                    "Data weights vector and dimension of data space does not match"
                )
            data_weights = data_weights / np.linalg.norm(data_weights)
            residuals = data_weights * residuals

        if method == "wme":
            qoi = np.sum(residuals, axis=1) / np.sqrt(sub_n_samples)
        elif method == "pca":
            # Learn qoi to use using PCA
            pca_res, X_train = pca(residuals, n_components=num_components)
            self.pca = {
                "X_train": X_train,
                "vecs": pca_res.components_,
                "var": pca_res.explained_variance_,
            }

            # Compute WME
            qoi = residuals @ pca_res.components_.T
        elif method == "svd":
            # Learn qoi to use using SVD
            u, s, v = svd(residuals)
            self.svd = {"u": u, "singular_values": s, "singular_vectors": v}
            qoi = residuals @ (v[0:num_components, :]).T
        else:
            ValueError(f"Unrecognized QoI Map type {method}")

        # qoi = qoi.reshape(sub_n_samples, -1)
        d = DensityProblem(lam, qoi, self.domain, weights=sample_weights)

        return d
