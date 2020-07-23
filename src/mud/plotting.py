from matplotlib import pyplot as plt
import numpy as np

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
    AA = np.hstack([sp.linalg.null_space(A[i,:].reshape(1,-1)) for i in range(numQoI)]).T
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


