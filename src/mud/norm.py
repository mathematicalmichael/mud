import numpy as np

def mynorm(X, mat):
    """"
    Inner-product induced vector norm implementation.
    `(x,x)_C := x^T C^-1 x` 
    `mat` is taken to be the covariance, the inverse is taken to form
    the precision matrix that will weight our inner-product space.
    """
    Y = (np.linalg.inv(mat) @ X)
    return np.sum(X * Y, axis=0)


def full_functional(operator, inputs, data, initial_mean, initial_cov, observed_mean=0, observed_cov=1):
    return norm_input(inputs, initial_mean, initial_cov) +\
            norm_data(operator, inputs, data, observed_mean, observed_cov) -\
            norm_predicted(operator, inputs, initial_mean, initial_cov)

def norm_input(inputs, initial_mean, initial_cov):
    X = (inputs - initial_mean.T).T
    return mynorm(X, initial_cov)


def norm_data(operator, inputs, data, observed_mean=0, observed_cov=1):
    if isinstance(observed_cov, int) or isinstance(observed_cov, float):
        observed_cov = observed_cov*np.eye(len(data))
    X = (operator@inputs.T + data) - observed_mean.T
    return mynorm(X, observed_cov)


def norm_predicted(operator, inputs, initial_mean, initial_cov):
    predicted_cov = operator@initial_cov@operator.T
    # since the operator is affine, we can factor it (for efficiency) as if it were linear
    X = operator@(inputs-initial_mean.T).T
    return mynorm(X, predicted_cov)


#def functional(mu_init, sigma_init, A, b, lam, PF_cov, observed_mean):
#    return np.linalg.norm(np.linalg.inv(sigma_init)@(lam-mu_init))**2 +\
#           np.linalg.norm(A@lam+b - observed_mean)**2 -\
#           np.linalg.norm(np.linalg.inv(np.sqrt(PF_cov))@A@(lam-mu_init))**2


