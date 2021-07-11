import numpy as np


def inner_product(X, mat):
    """
    Inner-product induced vector norm implementation.

    Returns square of norm defined by the inner product
    ``(x,x)_C := x^T C^-1 x``

    Parameters
    ----------
    X : (M, N) array_like
        Input array. N = number of samples, M = dimension

    mat : (M, M) array_like
        Positive-definite operator which induces the inner product

    Returns
    -------
    Z : (N, 1) ndarray
        inner-product of each column in ``X`` with respect to ``mat``

    """
    Y = np.linalg.inv(mat) @ X
    result = np.sum(X * Y, axis=0)
    return result


def full_functional(
    operator, inputs, data, initial_mean, initial_cov, observed_mean=0, observed_cov=1
):
    return (
        norm_input(inputs, initial_mean, initial_cov)
        + norm_data(operator, inputs, data, observed_mean, observed_cov)
        - norm_predicted(operator, inputs, initial_mean, initial_cov)
    )


def norm_input(inputs, initial_mean, initial_cov):
    if isinstance(initial_cov, int) or isinstance(initial_cov, float):
        initial_cov = initial_cov * np.eye(len(initial_mean))
    X = (inputs - initial_mean.T).T
    return inner_product(X, initial_cov)


def norm_data(operator, inputs, data, observed_mean, observed_cov):
    if isinstance(observed_cov, int) or isinstance(observed_cov, float):
        observed_cov = observed_cov * np.eye(len(data))
    X = (operator @ inputs.T + data) - observed_mean.T
    return inner_product(X, observed_cov)


def norm_predicted(operator, inputs, initial_mean, initial_cov):
    if isinstance(initial_cov, int) or isinstance(initial_cov, float):
        initial_cov = initial_cov * np.eye(len(initial_mean))
    predicted_cov = operator @ initial_cov @ operator.T
    # if operator is affine, we can factor it (for efficiency) to be same as linear
    X = operator @ (inputs - initial_mean.T).T
    return inner_product(X, predicted_cov)
