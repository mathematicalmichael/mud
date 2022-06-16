import pdb

# Plotting libraries
import matplotlib.pyplot as plt
import numpy as np

# Mud libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(data, n_components=2):
    """
    Apply PCA to Time-Series Data-set DS


    Parameters
    ----------
    ds : ArrayLike
        Data to apply PCA transformation to.
    n_components: int, default=2
        Number of principal components to use.
    """
    # Standarize and perform linear PCA
    sc = StandardScaler()
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(sc.fit_transform(data))

    return pca, X_train


def plot_pca_trained_ds(X_train, ax=None, label=True, s=1):
    # Plot density of trained data
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], s=s)

    if label:
        ax.set_title("$\\boldsymbol{ T_2 = XW_2 }$")
        ax.set_xlabel("$t_1$")
        ax.set_ylabel("$t_2$")

    return ax


def plot_pca_vecs(pca, ax=None, fixed=True, lims=None, label=True):
    # Plot density of trained data
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    # Plot first two principle component vectors.
    vec1 = pca.components_[0]
    vec2 = pca.components_[1]
    if fixed:
        vec1 = vec1 if vec1[0] > 0 else -1.0 * vec1
        vec1 = vec1 / np.max(vec1)
        vec2 = vec2 if vec2[0] < 0 else -1.0 * vec2
        vec2 = vec2 / np.max(vec2)

    plt.scatter(np.arange(len(vec1)), vec1, s=1, label="$\\boldsymbol{w}_0$")

    # ax2 = ax.twinx()
    plt.scatter(
        np.arange(len(vec2)), vec2, s=1, color="orange", label="$\\boldsymbol{w}_1$"
    )

    if lims:
        ax.set_ylim(lims[0])
        # ax2.set_ylim(lims[1])

    if label:
        ordinal = lambda n: "%d%s" % (
            n,
            "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )
        ax.set_title("$\\boldsymbol{w}_i$")
        ax.set_xlabel("index")
        ax.set_ylabel("$(\\boldsymbol{w}_0)_i$")
        # ax2.set_ylabel("$(\\boldsymbol{w}_1)_i$")

        h1, l1 = ax.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        # ax.legend(h1+h2, l1+l2, loc='lower left', fontsize=14)
        ax.legend(h1, l1, loc="lower left")

    return ax


def plot_pca_sample_contours(ds, X_train, ax=None, i=0, s=100, label=True):
    # Plot density of trained data
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    # Plot lambda spaced contoured by our trained PCA data set.
    plt.scatter(ds["lam"][:, 0], ds["lam"][:, 1], c=X_train[:, i], s=s)

    if label:
        if i == 0:
            ax.set_title("$t_{i,1} = x_{i,j}w_{j,1}$")
        else:
            ax.set_title("$t_{i,2} = x_{i,j}w_{j,2}$")
        ax.set_xlabel("$\\Lambda_1$")
        ax.set_ylabel("$\\Lambda_2$")

    return ax


