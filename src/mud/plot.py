from matplotlib import pyplot as plt
import numpy as np

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

    >>> from scipy.linalg import null_space
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

def plotChain(mud_chain, ref_param, color='k'):
    num_steps = len(mud_chain)
    current_point = mud_chain[0]
    plt.scatter(current_point[0], current_point[1], c='b')
    for i in range(0,num_steps):
        next_point = mud_chain[i]
        points = np.hstack([current_point, next_point])
        plt.plot(points[0,:], points[1,:], c=color)
        current_point = next_point

    plt.ylim([0,1])
    plt.xlim([0,1])
#     plt.axis('off')
    plt.scatter(ref_param[0], ref_param[1], c='r')


def plot_contours(A,color='k'):
    numQoI = A.shape[0]
    AA = np.hstack([null_space(A[i,:].reshape(1,-1)) for i in range(numQoI)]).T
    for i in range(numQoI):
        plt.plot([0.5-AA[i,0],0.5+AA[i,0]], [0.5-AA[i,1], 0.5+AA[i,1]], c=color, ls=':')


def make_2d_normal_mesh(N=50, window=1):
    X = np.linspace(-window, window, N)
    Y = np.linspace(-window, window, N)
    X, Y = np.meshgrid(X, Y)
    XX = np.vstack([X.ravel(), Y.ravel()]).T
    return (X, Y, XX)

def make_2d_unit_mesh(N=50, window=1):
    X = np.linspace(0, window, N)
    Y = np.linspace(0, window, N)
    X, Y = np.meshgrid(X, Y)
    XX = np.vstack([X.ravel(), Y.ravel()]).T
    return (X, Y, XX)


