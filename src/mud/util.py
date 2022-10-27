from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erfinv  # type: ignore


def std_from_equipment(tolerance=0.1, probability=0.95):
    """
    Converts tolerance `tolerance` for precision of measurement
    equipment to a standard deviation, scaling so that
    (100`probability`) percent of measurements are within `tolerance`.
    A mean of zero is assumed. `erfinv` is imported from `scipy.special`
    """
    standard_deviation = tolerance / (erfinv(probability) * np.sqrt(2))
    return standard_deviation


def transform_linear_map(operator, data, std):
    """
    Takes a linear map `operator` of size (len(data), dim_input)
    or (1, dim_input) for repeated observations, along with
    a vector `data` representing observations. It is assumed
    that `data` is formed with `M@truth + sigma` where `sigma ~ N(0, std)`

    This then transforms it to the MWE form expected by the DCI framework.
    It returns a matrix `A` of shape (1, dim_input) and np.float `b`
    and transforms it to the MWE form expected by the DCI framework.

    >>> X = np.ones((10, 2))
    >>> x = np.array([0.5, 0.5]).reshape(-1, 1)
    >>> std = 1
    >>> d = X @ x
    >>> A, b = transform_linear_map(X, d, std)
    >>> np.linalg.norm(A @ x + b)
    0.0
    >>> A, b = transform_linear_map(X, d, [std]*10)
    >>> np.linalg.norm(A @ x + b)
    0.0
    >>> A, b = transform_linear_map(np.array([[1, 1]]), d, std)
    >>> np.linalg.norm(A @ x + b)
    0.0
    >>> A, b = transform_linear_map(np.array([[1, 1]]), d, [std]*10)
    Traceback (most recent call last):
    ...
    ValueError: For repeated measurements, pass a float for std
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


def transform_linear_setup(operator_list, data_list, std_list):
    if isinstance(std_list, (float, int)):
        std_list = [std_list] * len(data_list)
    # repeat process for multiple quantities of interest
    results = [
        transform_linear_map(o, d, s)
        for o, d, s in zip(operator_list, data_list, std_list)
    ]
    operators = [r[0] for r in results]
    datas = [r[1] for r in results]
    return np.vstack(operators), np.vstack(datas)


def null_space(A, rcond=None):
    """
    Construct an orthonormal basis for the null space of A using SVD

    Method is slight modification of ``scipy.linalg``

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Z : (N, K) ndarray
        Orthonormal basis for the null space of A.
        K = dimension of effective null space, as determined by rcond


    Examples
    --------
    One-dimensional null space:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [1, 1]])
    >>> ns = null_space(A)
    >>> ns * np.sign(ns[0,0])  # Remove the sign ambiguity of the vector
    array([[ 0.70710678],
           [-0.70710678]])

    Two-dimensional null space:

    >>> B = np.random.rand(3, 5)
    >>> Z = null_space(B)
    >>> Z.shape
    (5, 2)
    >>> np.allclose(B.dot(Z), 0)
    True

    The basis vectors are orthonormal (up to rounding error):

    >>> np.allclose(Z.T.dot(Z), np.eye(2))
    True

    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


def make_2d_unit_mesh(N: int = 50, window: int = 1):
    """
    Make 2D Unit Mesh

    Constructs mesh based on uniform distribution to discretize each axis.

    Parameters
    ----------
    N : int, default=50
        Size of unit mesh. `N` points will be generated in each x,y direction.
    window : int, default=1
        Upper bound of mesh. Lower bound fixed at 0 always.

    Returns
    -------
    grid : tuple of np.ndarray
        Tuple of `(X, Y, XX)`, the grid `X` and `Y` and 2D mesh `XX`

    Example Usage
    -------------

    >>> x, y, XX = make_2d_unit_mesh(3)
    >>> print(XX)
    [[0.  0. ]
     [0.5 0. ]
     [1.  0. ]
     [0.  0.5]
     [0.5 0.5]
     [1.  0.5]
     [0.  1. ]
     [0.5 1. ]
     [1.  1. ]]
    """
    X = np.linspace(0, window, N)
    Y = np.linspace(0, window, N)
    X, Y = np.meshgrid(X, Y)
    XX = np.vstack([X.ravel(), Y.ravel()]).T
    return (X, Y, XX)


def add_noise(signal: ArrayLike, sd: float = 0.05, seed: int = None):
    """
    Add Noise

    Add noise to synthetic signal to model a real measurement device. Noise is
    assumed to be from a standard normal distribution std deviation `sd`:

    $\\mathcal{N}(0,\\sigma)$

    Paramaters
    ---------
    signal : numpy.typing.ArrayLike
      Signal to add noise to.
    sd : float, default = 0.05
      Standard deviation of error to add.
    seed : int, optional
      Seed to use for numpy random number generator.

    Returns
    -------
    noisy_signal: numpy.typing.ArrayLike
      Signal with noise added to it.

    Example Usage
    -------------
    Generate test signal, add noise, check average distance
    >>> seed = 21
    >>> test_signal = np.ones(5)
    >>> noisy_signal = add_noise(test_signal, sd=0.05, seed=21)
    >>> np.round(1000*np.mean(noisy_signal-test_signal))
    4.0
    """
    signal = np.array(signal)

    if seed is not None:
        np.random.seed(seed)

    # Populate qoi_true with noise
    noise = np.random.randn(signal.size) * sd

    return signal + noise


def rank_decomposition(A: np.typing.ArrayLike) -> List[np.ndarray]:
    """Build list of rank k updates of A"""
    A = np.array(A)
    A_ranks = []
    rank_1_updates = []
    u, s, v = np.linalg.svd(A)
    A_ranks.append(s[0] * (u[:, 0].reshape(-1, 1)) @ v[:, 0].reshape(1, -1))
    for i in range(1, A.shape[1]):
        rank_1_updates.append(s[i] * (u[:, i].reshape(-1, 1)) @ v[:, i].reshape(1, -1))
        A_ranks.append(sum(rank_1_updates[0:i]))

    return A_ranks


def fit_domain(
    x: np.ndarray = None, min_max_bounds: np.ndarray = None, pad_ratio: float = 0.1
) -> np.ndarray:
    """
    Fit domain bounding box to array x

    Parameters
    ----------
    x : ArrayLike
        2D array to calculate min, max values along columns.
    pad_ratio : float, default=0.1
        What ratio of total range=max-min to subtract/add to min/max values to
        construct final domain range. Padding is done per x column dimension.

    Returns
    -------
    min_max_bounds : ArrayLike
        Domain fitted to values found in 2D array x, with padding added.

    Examples
    --------
    Input must be 2D. Set pad_ratio = 0 to get explicit min/max bounds
    >>> fit_domain(np.array([[1, 10], [0, -10]]), pad_ratio=0.0)
    array([[  0,   1],
           [-10,  10]])

    Can extend domain around the array values using the pad_ratio argument.

    >>> fit_domain(np.array([[1, 10], [0, -10]]), pad_ratio=1)
    array([[ -1,   2],
           [-30,  30]])
    """
    if min_max_bounds is None:
        if x is None:
            raise ValueError("Both x and min_max_bounds can't be None")
        min_max_bounds = np.array([x.min(axis=0), x.max(axis=0)]).T
    pad = pad_ratio * (min_max_bounds[:, 1] - min_max_bounds[:, 0])
    min_max_bounds[:, 0] = min_max_bounds[:, 0] - pad
    min_max_bounds[:, 1] = min_max_bounds[:, 1] + pad
    return min_max_bounds


def set_shape(array: np.ndarray, shape: Union[List, Tuple] = (1, -1)) -> np.ndarray:
    """Resizes inputs if they are one-dimensional."""
    return array.reshape(shape) if array.ndim < 2 else array
