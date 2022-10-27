"""
Exponential Decay Example


"""
import numpy as np
from scipy.stats.distributions import uniform  # type: ignore

from mud.base import SpatioTemporalProblem


def exp_decay_1D(
    u_0=0.75,
    time_range=[0, 4.0],
    domain=[0, 1],
    num_samples=10000,
    lambda_true=0.5,
    t_start=0.0,
    sampling_freq=100.0,
    std_dev=0.05,
):
    def u_t_lambda_1(t, l1):
        return u_0 * np.exp(-np.outer(l1, t))

    # Build initial samples
    initial = uniform(loc=domain[0], scale=domain[1] - domain[0])

    exp_decay = SpatioTemporalProblem()
    exp_decay.domain = domain
    exp_decay.times = np.arange(t_start, time_range[1], 1 / sampling_freq)
    exp_decay.sample_dist = "u"
    exp_decay.lam = initial.rvs(size=num_samples)
    exp_decay.data = u_t_lambda_1(exp_decay.times, exp_decay.lam)
    exp_decay.true_vals = u_t_lambda_1(exp_decay.times, lambda_true)[0]
    exp_decay.std_dev = std_dev

    return exp_decay


def exp_decay_2D(
    time_range=[0, 3.0],
    domain=np.array([[0.7, 0.8], [0.25, 0.75]]),
    num_samples=100,
    lambda_true=[0.75, 0.5],
    N=100,
    t_start=0.0,
    sampling_freq=10.0,
    std_dev=0.05,
):
    def u_t_lambda_2(t, l1, l2):
        return (l1 * np.exp(-np.outer(t, l2))).T

    # Build initial samples
    num_params = domain.shape[0]
    mn = np.min(domain, axis=1)
    mx = np.max(domain, axis=1)
    initial = uniform(loc=mn, scale=mx - mn)

    exp_decay = SpatioTemporalProblem()
    exp_decay.domain = domain
    exp_decay.times = np.arange(t_start, time_range[1], 1 / sampling_freq)[0:N]
    exp_decay.sample_dist = "u"
    exp_decay.lam = initial.rvs(size=(num_samples, num_params))
    exp_decay.data = u_t_lambda_2(
        exp_decay.times, exp_decay.lam[:, 0], exp_decay.lam[:, 1]
    )
    exp_decay.true_vals = u_t_lambda_2(exp_decay.times, lambda_true)[0]
    exp_decay.std_dev = std_dev

    return exp_decay
