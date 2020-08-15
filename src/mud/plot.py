from matplotlib import pyplot as plt
import numpy as np
from mud.util import null_space

def plotChain(mud_chain, ref_param, color='k', s=100):
    num_steps = len(mud_chain)
    current_point = mud_chain[0]
    plt.scatter(current_point[0], current_point[1], c='b', s=s)
    for i in range(0,num_steps):
        next_point = mud_chain[i]
        points = np.hstack([current_point, next_point])
        plt.plot(points[0,:], points[1,:], c=color)
        current_point = next_point

    plt.ylim([0,1])
    plt.xlim([0,1])
#     plt.axis('off')
    plt.scatter(ref_param[0], ref_param[1], c='r', s=s)


def plot_contours(A, ref_param, subset=None,
                  color='k', ls=':', lw=1, fs=20, w=1, s=100, **kwds):
    if subset is None: subset = np.arange(A.shape[0])
    A = A[np.array(subset),:]
    numQoI = A.shape[0]
    AA = np.hstack([null_space(A[i,:].reshape(1,-1)) for i in range(numQoI)]).T
    for i, contour in enumerate(subset):
        xloc = [ref_param[0] - w*AA[i,0], ref_param[1] + w*AA[i,0]]
        yloc = [ref_param[0] - w*AA[i,1], ref_param[1] + w*AA[i,1]]
        plt.plot(xloc, yloc, c=color, ls=ls, lw=lw, **kwds)
        plt.annotate('%d'%(contour+1), (xloc[0], yloc[0]), fontsize=fs)


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


