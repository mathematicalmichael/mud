# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = mud.mud:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys
import logging
import numpy as np

from mud import __version__

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Demonstration of analytical MUD point")
    parser.add_argument(
        "--version",
        action="version",
        version="mud {ver}".format(ver=__version__))
    parser.add_argument(
        dest="n",
        help="Number of QoI",
        type=int,
        metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("Using {} Quantities of Interest".format(args.n))
    _logger.info("Script end.")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])

############################################################
def makeRi(A, initial_cov):
    predicted_cov = A@initial_cov@A.T
    if isinstance(predicted_cov, float):
        ipc = 1./predicted_cov * np.eye(1)
    else:
        ipc = np.linalg.inv(predicted_cov)
    Ri = np.linalg.inv(initial_cov) - A.T@ ipc@ A
    return Ri


def mud_sol(A, b, y=None, mean=None, cov=None, data_cov=None):
    """
    Definitely works.
    For SWE problem, we are inverting N(0,1).
    This is the defautl value for `data_cov`.
    """
    if data_cov is None: data_cov = np.eye(A.shape[0])
    if cov is None: cov = np.eye(A.shape[1])
    if mean is None: mean = np.zeros((A.shape[1],1))
    if y is None: y = np.zeros((A.shape[0],1))

    z = y.ravel() - b.ravel() - (A@mean).ravel()
    z = z.reshape(-1,1)

    # compute once for re-use
    pre = A@cov@A.T
    ipc = np.linalg.pinv(pre)

    update = cov@A.T@ipc
    mud_point = mean.ravel() + (update@z).ravel()
    return mud_point.reshape(-1,1)


def mud_sol_alt(A, b, y=None, mean=None, cov=None, data_cov=None):
    """
    Doesn't use R directly, uses new equations.
    This presents the equation as a rank-k update
    to the error of the initial estimate.
    """
    if data_cov is None: data_cov = np.eye(A.shape[0])
    if cov is None: cov = np.eye(A.shape[1])
    if mean is None: mean = np.zeros((A.shape[1],1))
    if y is None: y = np.zeros((A.shape[0],1))

    z = y.ravel() - b.ravel() - (A@mean).ravel()
    z = z.reshape(-1,1)

    # compute once for re-use
    idc = np.linalg.inv(data_cov)
    pred_cov = A@cov@A.T
    ipc = np.linalg.pinv(pred_cov)
    # pinv b/c inv unstable for rank-deficient A
    
    # Form derived via Hua's identity + Woodbury
    up_cov = cov - cov@A.T@ipc@(pred_cov - data_cov)@ipc@A@cov
    update = up_cov @ A.T @ idc
    mud_point = mean.ravel() + (update @ z).ravel()

    # mud_point = mean.ravel() + (cov@A.T@ipc@x).ravel()
    return mud_point.reshape(-1,1)


def map_sol(A, b, y=None, mean=None, cov=None, data_cov=None, w=1):
    if data_cov is None: data_cov = np.eye(A.shape[0])
    if cov is None: cov = np.eye(A.shape[1])
    if mean is None: mean = np.zeros((A.shape[1],1))
    if y is None: y = np.zeros((A.shape[0],1))

    z = y.ravel() - b.ravel() - (A@mean).ravel()
    z = z.reshape(-1,1)

    precision = np.linalg.inv(A.T@np.linalg.inv(data_cov)@A + w*np.linalg.inv(cov))
    update = precision@A.T@np.linalg.inv(data_cov)
    map_point = mean.ravel() + (update @ z).ravel()
    return map_point.reshape(-1,1)


def performEpoch(A, b, y, initial_mean, initial_cov, data_cov=None, idx=None):
    dim_out = A.shape[0]
    mud_chain = []

    current_mean = initial_mean
    mud_chain.append(current_mean)
    if idx is None: idx = range(dim_out)
    for i in idx:
        _A = A[i,:].reshape(1,-1)
        _b = b[i]
        _y = y[i]
        mud_chain.append(mud_sol(_A, _b, _y, current_mean, initial_cov, data_cov=None))
        current_mean = mud_chain[-1]
    return mud_chain


def iterate(A, b, y, initial_mean, initial_cov, data_cov=None, num_epochs=1, idx=None):
    mud_chain = performEpoch(A, b, y, initial_mean, initial_cov, data_cov, idx)
    for _ in range(1, num_epochs):
        mud_chain += performEpoch(A, b, y, mud_chain[-1], initial_cov, data_cov, idx)

    return mud_chain


if __name__ == "__main__":
    run()
