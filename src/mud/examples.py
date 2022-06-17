import numpy as np
from matplotlib import pyplot as plt
from mud.base import DensityProblem, IterativeLinearProblem
from mud.funs import wme
from mud.util import std_from_equipment
from scipy.stats import distributions as ds


def rotation_map(qnum=10, tol=0.1, b=None, ref_param=None, seed=None):
    """
    Generate test data linear rotation map

    """
    if seed is not None:
        np.random.seed(seed)

    vec = np.linspace(0, np.pi, qnum)
    A = np.array([[np.sin(theta), np.cos(theta)] for theta in vec])
    A = A.reshape(qnum, 2)
    b = np.zeros((qnum, 1)) if b is None else b
    ref_param = (
        np.array([[0.5, 0.5]]).reshape(-1, 1) if ref_param is None else ref_param
    )

    # Compute observed value
    y = A @ ref_param + b
    initial_mean = np.random.randn(2).reshape(-1, 1)
    initial_cov = np.eye(2) * std_from_equipment(tol)

    return (A, b, y, initial_mean, initial_cov, ref_param)


def rotation_map_trials(
    numQoI=10,
    method="ordered",
    num_trials=100,
    model_eval_budget=100,
    ax=None,
    color="r",
    label="Ordered QoI $(10\\times 10D)$",
    seed=None,
):
    """
    Run a set of trials for linear rotation map problems

    """

    # Initialize plot if axis object is not passed in
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Build Rotation Map. This will initialize seed of trial if specified
    A, b, y, initial_mean, initial_cov, ref_param = rotation_map(qnum=numQoI, seed=seed)

    # Calcluate number of epochs per trial using budget and number of QoI
    num_epochs = model_eval_budget // numQoI

    errors = []
    for trial in range(num_trials):
        # Get a new random initial mean to start from per trial on same problem
        initial_mean = np.random.rand(2, 1)

        # Initialize number of epochs and idx choices to use on this trial
        epochs = num_epochs
        choice = np.arange(numQoI)

        # Modify epochs/choices based off of method
        if method == "ordered":
            # Ordered - GO through each row in order once per epoch, each trial
            epochs = num_epochs
        elif method == "shuffle":
            # Shuffled - Shuffle rows on each trial, in order once per epoch
            np.random.shuffle(choice)
            epochs = num_epochs
        elif method == "batch":
            # Batch - Perform only one epoch, but iterate in random
            #         order num_epochs times over each row of A
            choice = list(np.arange(numQoI)) * num_epochs
            np.random.shuffle(choice)
            epochs = 1
        elif method == "random":
            # Randoms - Perform only one epoch, but do num_epochs*rows
            #           random choices of rows of A, with replacement
            choice = np.random.choice(np.arange(numQoI), size=num_epochs * numQoI)
            epochs = 1

        # Initialize Iterative Linear Problem and solve using number of epochs
        prob = IterativeLinearProblem(
            A, b=b, y=y, initial_mean=initial_mean, cov=initial_cov, idx_order=choice
        )
        _ = prob.solve(num_epochs=epochs)

        # Plot errors with respect to reference parameter over each iteration
        prob.plot_chain_error(ref_param, alpha=0.1, ax=ax, color=color, fontsize=36)

        # Append to erros matrix to calculate mean error accross trials
        errors.append(prob.get_errors(ref_param))

    # Compute mean errors at each iteration across all trials
    avg_errs = np.mean(np.array(errors), axis=0)

    # Plot mean errors
    ax.plot(avg_errs, color, lw=5, label=label)


def identity_uniform_1D_density_prob(
    num_samples=2000,
    num_obs=20,
    y_true=0.5,
    noise=0.05,
    weights=None,
    domain=[0, 1],
    wme_map=True,
    analytical_pred=True,
):
    """
    1D Density Problem using WME on identity map with uniform initial

    Sets up a Density Problem using a given domain (unit by default) and a
    uniform initial distribution under an identity map and the Weighted
    Mean Error map to . This function is used
    as a set-up for tejjsts to the DensityProblem class.

    `num_obs` observations
    are collected from an initial distribution and used as the true signal,
    with noise being added to each observation.
    Sets up an inverse problem using the unit domain and uniform distribution
    under an identity map. This is equivalent to studying a
    \"steady state\" signal over time, or taking repeated measurements
    of the same quantity to reduce variance in the uncertainty.
    """
    init_dist = ds.uniform(loc=domain[0], scale=domain[1] - domain[0])
    X = init_dist.rvs(size=(num_samples, 1))

    if wme_map:
        y_pred = np.repeat(X, num_obs, 1)
        # data is truth + noise
        y_observed = y_true * np.ones(num_obs) + noise * np.random.randn(num_obs)
        Y = wme(y_pred, y_observed, sd=noise)

        # Build Density problem of M(X) = WME(X,y_observed) over domain
        D = DensityProblem(X, Y, np.array([domain]), weights=weights)
        D.set_initial(init_dist)

        if analytical_pred:
            # analytical construction of predicted domain under identity map.
            y_domain = np.repeat(np.array([[0], [1]]), num_obs, 1)
            mn, mx = wme(y_domain, y_observed, sd=noise)
            loc, scale = mn, mx - mn
            D.set_predicted(ds.uniform(loc=loc, scale=scale))
    else:
        # Build Density problem of M(X) = X over domain
        D = DensityProblem(X, X, np.array([domain]), weights=weights)
        D.set_initial(init_dist)

        if analytical_pred:
            D.set_predicted(init_dist)

    return D
