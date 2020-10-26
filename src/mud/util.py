import numpy as np
from pyerf import erfinv

def std_from_equipment(tolerance=0.1, probability=0.95):
    """
    Converts tolerance `tolerance` for precision of measurement
    equipment to a standard deviation, scaling so that
    (100`probability`) percent of measurements are within `tolerance`.
    A mean of zero is assumed. `erfinv` is imported from `scipy.special`
    """
    standard_deviation = tolerance/(erfinv(probability)*np.sqrt(2))
    return standard_deviation


def rotationMap(qnum = 10, orth=True):
    if orth:
        return np.array([[np.sin(theta), np.cos(theta)] for theta in np.linspace(0, np.pi, qnum+1)[0:-1]]).reshape(qnum,2)
    else:
        return np.array([[np.sin(theta), np.cos(theta)] for theta in np.linspace(0, np.pi, qnum)]).reshape(qnum,2)


def transform_linear_map(operator, data, std):
    """
    Takes a linear map `operator` of size (len(data), dim_input)
    and transforms it to the MWE form expected by the DCI framework.
    """
    num_observations = len(data)
    assert operator.shape[0] == num_observations, "Operator shape mismatch"
    if isinstance(std, int) or isinstance(std, float):
        std = np.array([std]*num_observations)
    if isinstance(std, list) or isinstance(std, tuple):
        std = np.array(std)
    if isinstance(data, np.ndarray):
        data = list(data.ravel())
    assert len(std) == num_observations, "Standard deviation shape mismatch"
    D = np.diag(1./(std*np.sqrt(num_observations)))
    A = np.sum(D@operator, axis=0)
    b = np.sum(np.divide(data, std))
    return A, (-1.0/np.sqrt(num_observations))*b.reshape(-1,1)


def transform_setup(operator_list, data_list, std_list):
    # repeat process for multiple quantities of interest
    results   = [transform_linear_map(o, d, s) for o,d,s in zip(operator_list, data_list, std_list)]
    operators = [r[0] for r in results]
    datas     = [r[1] for r in results]
    return np.vstack(operators), np.vstack(datas)


def createRandomLinearMap(dim_input, dim_output, dist='normal', repeated=False):
    """
    Create random linear map from P dimensions to S dimensions.
    """
    if  dist == 'normal':
        M     = np.random.randn(dim_output, dim_input)
    else:
        M     = np.random.rand(dim_output, dim_input)
    if repeated: # just use first row
        M     = np.array(list(M[0,:])*dim_output).reshape(dim_output, dim_input)

    return M


def createNoisyReferenceData(M, reference_point, std):
    dim_input  = len(reference_point)
    dim_output = M.shape[0]
    assert M.shape[1] == dim_input, "Mperator/Data dimension mismatch"
    if isinstance(std, int) or isinstance(std, float):
        std    = np.array([std]*dim_output)

    ref_input  = np.array(list(reference_point)).reshape(-1,1)
    ref_data   = M@ref_input
    noise      = np.diag(std)@np.random.randn(dim_output,1)
    data       = ref_data + noise
    return data


def createRandomLinearPair(reference_point, num_observations, std,
                          dist='normal', repeated=False):
    """
    data will come from a normal distribution centered at zero
    with standard deviation given by `std`
    QoI map will come from standard uniform or normal if dist=normal
    if `repeated` is True, the map will be rank 1.
    """
    dim_input = len(reference_point)
    M         = createRandomLinearMap(dim_input, num_observations, dist, repeated)
    data      = createNoisyReferenceData(M, reference_point, std)
    return M, data


def createRandomLinearProblem(reference_point, num_qoi,
                              num_observations_list, std_list,
                              dist='normal', repeated=False):
    """
    Wrapper around `createRandomLinearQoI` to generalize to multiple QoI maps.
    """
    if isinstance(std_list, int) or isinstance(std_list, float):
        std_list                = [std_list]*num_qoi
    else:
        assert len(std_list) == num_qoi

    if isinstance(num_observations_list, int) or isinstance(num_observations_list, float):
        num_observations_list   = [num_observations_list]*num_qoi
    else:
        assert len(num_observations_list) == num_qoi

    assert len(std_list) == len(num_observations_list)
    results       = [createRandomLinearPair(reference_point, n, s, dist, repeated) \
                     for n,s in zip(num_observations_list, std_list)]
    operator_list = [r[0] for r in results]
    data_list     = [r[1] for r in results]
    return operator_list, data_list, std_list


def null_space(A, rcond=None):
    """
    Construct an orthonormal basis for the null space of A using SVD

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

    See also
    --------
    svd : Singular value decomposition of a matrix
    orth : Matrix range

    Examples
    --------
    One-dimensional null space:

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

    >>> Z.T.dot(Z)
    array([[  1.00000000e+00,   6.92087741e-17],
           [  6.92087741e-17,   1.00000000e+00]])

    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q
