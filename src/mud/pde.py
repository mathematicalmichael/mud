# Iterative Mud Notebook
import logging
import pdb
import pickle
from datetime import datetime
from pathlib import Path

# Plotting libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from IPython.display import Image
import mud.pca as pca
from mud.base import DensityProblem
from mud.funs import iterative_mud_problem, wme

# Mud libraries
from mud.util import add_noise
from pyadcirc.io import read_fort222
from pyadcirc.viz import generate_gif as gg
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde
from scipy.stats import norm, uniform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

_logger = logging.getLogger(__name__)


class PDEProblem(object):
    """
    Class for parameter estimation problems for partial differential
    equation models of real world systems. Uses a QoI map of weighted
    residuals between simulated data and measurements to do inversion

    Attributes
    ----------
    TODO: Finish

    Methods
    -------
    TODO: Finish


    """

    def __init__(self, fname=None):

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

        if fname is not None:
            self.load(fname)

    @property
    def n_samples(self):
        if self.lam is None:
            raise AttributeError("lambda not yet set.")
        return self.lam.shape[0]

    @property
    def n_qoi(self):
        if self.qoi is None:
            raise AttributeError("qoi not yet set.")
        return self.qoi.shape[1]

    @property
    def n_sensors(self):
        if self.sensors is None:
            raise AttributeError("sensors not yet set.")
        return self.sensors.shape[0]

    @property
    def n_ts(self):
        if self.times is None:
            raise AttributeError("times not yet set.")
        return self.times.shape[0]

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
            if shape[0] != self.lam.shape[1]:
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
            # Expected to be in (# sampes x # sensors # # timesteps)
            data = np.reshape(data, (dim[0], -1))

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
                    "Dimensions of simulated data does not match number of (timesteps x sensors)"
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

    def measurements_from_reference(self, ref=None, std_dev=None):
        """
        Add noise to a reference solution.
        """
        if ref is not None:
            self._true_vals = ref
        if std_dev is not None:
            self.std_dev = std_dev
        if self.true_vals is None or self.std_dev is None:
            raise AttributeError('Must set reference solution and std_dev first or pass as arguments.')
        self.measurements = add_noise(self.true_vals, self.std_dev)

    def load(
        self,
        fname,
        lam="lam",
        data="data",
        true_vals=None,
        measurements=None,
        std_dev=None,
        sample_dist=None,
        domain=None,
        lam_ref=None,
        sensors=None,
        time=None,
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
        data : dict,
            Dictionary containing data from file for PDE problem class

        """
        _logger.info(f"Attempting to load {fname} from disk")
        try:
            if fname.endswith("nc"):
                _logger.info("Loading netcdf file")
                ds = xr.load_dataset(fname)
            else:
                _logger.info("Loading pickle file")
                with open(fname, "rb") as fp:
                    ds = pickle.load(fp)
        except FileNotFoundError:
            _logger.info(f"Failed to load {fname} from disk")
            raise FileNotFoundError(f"Couldn't find PDEProblem class data")

        get_set_val = lambda x: ds[x] if type(x) == str else x

        if sensors is not None:
            self.sensors = get_set_val(sensors)
        if time is not None:
            self.time = get_set_val(time)
        if domain is not None:
            self.domain = get_set_val(domain)
        if lam_ref is not None:
            self.domain = get_set_val(lam_ref)
        if measurements is not None:
            self.domain = get_set_val(measurements)

        self.lam = get_set_val(lam)
        self.data = get_set_val(data)

    def validate(
            self,
            check_meas=True,
            check_true=False,
    ):
        """Validates if class has been set-up appropriately for inversion"""
        req_attrs = ['domain','lam','data']
        if check_meas:
            req_attrs.append('measurements')
        if check_ref:
            req_attrs.append('true_lam')
            req_attrs.append('true_vals')

        missing = [x for x in req_attrs if self.__getattribute__(x)==None]
        if len(missing) > 0:
            raise ValueError(f'Missing attributes {missing}')


    def sample_data(
        self,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
        samples_idx=None,
        times_idx=None,
        sensors_idx=None,
    ):
        if self.data is None:
            raise AttributeError("data not set yet.")
        # Select data to plot
        sub_data = np.reshape(self.data, (self.n_samples, self.n_sensors, self.n_ts))
        sub_times = self.times
        sub_sensors = self.sensors

        if self.measurements is not None:
            sub_meas = np.reshape(self.measurements, (self.n_sensors, self.n_ts))
        else:
            sub_meas = None

        if times_mask is not None:
            sub_data = sub_data[:, :, times_mask]
            sub_times = sub_times[times_mask]
            if self.measurements is not None:
                sub_meas = sub_meas[:, times_mask]
        if times_idx is not None:
            times_idx = np.reshape(times_idx, (-1, 1))
            sub_data = sub_data[:, :, times_idx]
            sub_times = sub_times[times_idx]
            if self.measurements is not None:
                sub_meas = sub_meas[:, times_idx]
        if sensors_mask is not None:
            sub_data = sub_data[:, sensors_mask, :]
            sub_sensors = sub_sensors[sensors_mask]
            if self.measurements is not None:
                sub_meas = sub_meas[sensors_mask, :]
        if sensors_idx is not None:
            sensors_idx = np.reshape(sensors_idx, (-1, 1))
            sub_data = sub_data[:, sensors_idx, :]
            sub_sensors = sub_sensors[sensors_idx]
            if self.measurements is not None:
                sub_meas = sub_meas[sensors_idx, :]
        if samples_mask is not None:
            sub_data = sub_data[samples_mask, :, :]
        if samples_idx is not None:
            sub_data = sub_data[samples_idx, :, :]

        sub_data = np.reshape(
            sub_data, (self.n_samples, sub_times.shape[0] * sub_sensors.shape[0])
        )

        if self.measurements is not None:
            sub_meas = np.reshape(sub_meas, (len(sub_times) * len(sub_sensors)))

        return sub_times, sub_sensors, sub_data, sub_meas

    def plot_ts(
        self,
        ax=None,
        samples=None,
        times=None,
        sensor_idx=0,
        max_plot=100,
        alpha=0.1,
        fname=None,
        label=True,
    ):
        """
        Plot time series data
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 1, 1)

        times, _, sub_data, sub_meas = self.sample_data(
            samples_mask=samples, times_mask=times, sensors_idx=sensor_idx
        )
        num_samples = sub_data.shape[0]
        max_plot = num_samples if max_plot > num_samples else max_plot

        # Plot simulated data time series
        for i, idx in enumerate(np.random.choice(num_samples, max_plot)):
            if i != (max_plot - 1):
                _ = ax.plot(times, sub_data[i, :], "r-", alpha=alpha)
            else:
                _ = ax.plot(
                    times,
                    sub_data[i, :],
                    "r-",
                    alpha=alpha,
                    label=f"Sensor {sensor_idx}",
                )

        # Plot measured time series
        _ = plt.plot(times, sub_meas, "k^", label="$\\zeta_{obs}$", markersize=1)
        _ = ax.set_title("")

        return ax

    def mud_problem(
        self,
        method="wme",
        data_weights=None,
        sample_weights=None,
        pca_components=2,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
        samples_idx=None,
        times_idx=None,
        sensors_idx=None
    ):
        """Build QoI Map Using Data and Measurements"""

        # TODO: Finish sample data implimentation
        times, sensors, sub_data, sub_meas = self.sample_data(
            samples_mask=samples_mask, times_mask=times_mask, sensors_mask=sensors_mask,
            samples_idx=samples_idx, times_idx=times_idx, sensors_idx=sensors_idx
        )
        residuals = np.subtract(sub_data, sub_meas.T) / self.std_dev
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
            pca_res, X_train = pca.apply_pca(residuals, n_components=pca_components)
            self.pca = {"X_train": X_train, "vecs": pca_res.components_}

            # Compute WME - Note with two components
            qoi = np.array([np.sum(v * residuals, axis=1) for v in self.pca["vecs"]])
        else:
            ValueError(f"Unrecognized QoI Map type {method}")

        qoi = qoi.reshape(sub_n_samples, -1)
        d = DensityProblem(self.lam, qoi, self.domain, weights=sample_weights)

        return d


