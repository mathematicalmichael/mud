import numpy as np
from pyerf import erfinv


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
    and transforms it to the MWE form expected by the DCI framework.
    """
    num_observations = len(data)
    assert operator.shape[0] == num_observations, "Operator shape mismatch"
    if isinstance(std, int) or isinstance(std, float):
        std = np.array([std] * num_observations)
    if isinstance(std, list) or isinstance(std, tuple):
        std = np.array(std)
    if isinstance(data, np.ndarray):
        data = list(data.ravel())
    assert len(std) == num_observations, "Standard deviation shape mismatch"
    D = np.diag(1.0 / (std * np.sqrt(num_observations)))
    A = np.sum(D @ operator, axis=0)
    b = np.sum(np.divide(data, std))
    return A, (-1.0 / np.sqrt(num_observations)) * b.reshape(-1, 1)


def transform_linear_setup(operator_list, data_list, std_list):
    # repeat process for multiple quantities of interest
    results   = [transform_linear_map(o, d, s) for
                 o, d, s in zip(operator_list, data_list, std_list)]
    operators = [r[0] for r in results]
    datas     = [r[1] for r in results]
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
