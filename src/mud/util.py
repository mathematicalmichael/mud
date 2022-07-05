from typing import List, Optional, Tuple, Union

import numpy as np
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


def transform_linear_map(
    operator: np.ndarray,
    data: Union[np.ndarray, List[float], Tuple[float]],
    std: Union[np.ndarray, float, List[float], Tuple[float]],
):
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


def transform_linear_setup(
    operator_list: List[np.ndarray],
    data_list: Union[List[np.ndarray], Tuple[np.ndarray]],
    std_list: Union[
        float,
        np.ndarray,
        List[float],
        Tuple[float],
        Tuple[Tuple[float]],
        List[List[float]],
    ],
):
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


def null_space(A: np.ndarray, rcond: Optional[float] = None):
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
    >>> from mud.util import null_space
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
    window : int, defalut=1
        Upper bound of mesh. Lower bound fixed at 0 always.

    Returns
    ----------
    grid : tuple of np.ndarray
        Tuple of `(X, Y, XX)`, the grid `X` and `Y` and 2D mesh `XX`

    Example Usage
    -------------

    >>> from mud.util import make_2d_unit_mesh
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


def make_2d_normal_mesh(N: int = 50, window: int = 1):
    """
    Constructs mesh based on normal distribution to
    discretize each axis.
    >>> from mud.util import make_2d_normal_mesh
    >>> x, y, XX = make_2d_normal_mesh(3)
    >>> print(XX)
    [[-1. -1.]
     [ 0. -1.]
     [ 1. -1.]
     [-1.  0.]
     [ 0.  0.]
     [ 1.  0.]
     [-1.  1.]
     [ 0.  1.]
     [ 1.  1.]]
    """
    X = np.linspace(-window, window, N)
    Y = np.linspace(-window, window, N)
    X, Y = np.meshgrid(X, Y)
    XX = np.vstack([X.ravel(), Y.ravel()]).T
    return (X, Y, XX)


def set_shape(array: np.ndarray, shape: Union[List, Tuple] = (1, -1)) -> np.ndarray:
    """Resizes inputs if they are one-dimensional."""
    return array.reshape(shape) if array.ndim < 2 else array
