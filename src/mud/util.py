import numpy as np
from scipy.special import erfinv

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
        M     = np.rand.rand(dim_output, dim_input)
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


