import numpy as np

def mynorm(X, mat):
    """"
    Inner-product induced vector norm implementation. Returns square of norm.
    `(x,x)_C := x^T C^-1 x` 
    `mat` is taken to be the covariance, the inverse is taken to form
    the precision matrix that will weight our inner-product space.
    """
    Y = (np.linalg.inv(mat) @ X)
    result = np.sum(X * Y, axis=0)
    return result


def full_functional(operator, inputs, data, initial_mean, initial_cov, observed_mean=0, observed_cov=1):
    return norm_input(inputs, initial_mean, initial_cov) +\
            norm_data(operator, inputs, data, observed_mean, observed_cov) -\
            norm_predicted(operator, inputs, initial_mean, initial_cov)


def norm_input(inputs, initial_mean, initial_cov):
    if isinstance(initial_cov, int) or isinstance(initial_cov, float):
        initial_cov = initial_cov*np.eye(len(initial_mean))
    X = (inputs - initial_mean.T).T
    return mynorm(X, initial_cov)


def norm_data(operator, inputs, data, observed_mean, observed_cov):
    if isinstance(observed_cov, int) or isinstance(observed_cov, float):
        observed_cov = observed_cov*np.eye(len(data))
    X = (operator@inputs.T + data) - observed_mean.T
    return mynorm(X, observed_cov)


def norm_predicted(operator, inputs, initial_mean, initial_cov):
    if isinstance(initial_cov, int) or isinstance(initial_cov, float):
        initial_cov = initial_cov*np.eye(len(initial_mean))
    predicted_cov = operator@initial_cov@operator.T
    # since the operator is affine, we can factor it (for efficiency) as if it were linear
    X = operator@(inputs-initial_mean.T).T
    return mynorm(X, predicted_cov)
