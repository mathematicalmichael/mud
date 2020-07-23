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


def makeRi(A, initial_cov):
    predicted_cov = A@initial_cov@A.T
    if isinstance(predicted_cov, float):
        ipc = 1./predicted_cov * np.eye(1)
    else:
        ipc = np.linalg.inv(predicted_cov)
    Ri = np.linalg.inv(initial_cov) - A.T@ ipc@ A
    return Ri


def mud_sol(A, b, y, mean, cov, data_cov=None):
    if data_cov is None:
        # for SWE problem, we are inverting N(0,1).
        data_cov = np.eye(A.shape[0])
    x = y - b - A@mean
    
    Ri = makeRi(A, cov)
    predicted_cov = A@cov@A.T
    up = np.linalg.inv(A.T@np.linalg.inv(data_cov)@A + Ri)
    update = up @ A.T @ np.linalg.inv(data_cov)

    mud_point = mean.ravel() + (update @ x).ravel()
    return mud_point.reshape(-1,1)


def mud_sol_alt(A, b, y, mean, cov, data_cov=None):
    """
    Defintely works, doesn't use R
    """
    if data_cov is None:
        # for SWE problem, we are inverting N(0,1).
        data_cov = np.eye(A.shape[0])
    x = y - b - A@mean

    predicted_cov = A@cov@A.T
    update = cov @ A.T @ np.linalg.inv(predicted_cov)
    mud_point = mean.ravel() + (update @ x).ravel()
    return mud_point.reshape(-1,1)
 

def map_sol(A, b, y, mean, cov, data_cov=None, w=1):
    if data_cov is None:
        # for SWE problem, we are inverting N(0,1).
        data_cov = np.eye(A.shape[0])
    x= y - b - A@mean

    precision = np.linalg.inv(A.T@np.linalg.inv(data_cov)@A + w*np.linalg.inv(cov))
    update = precision@A.T@np.linalg.inv(data_cov)
    map_point = mean.ravel() + (update @ x).ravel()
    return map_point.reshape(-1,1)


def performEpoch(A, b, y, initial_mean, initial_cov, data_cov=None, idx=None):
    dim_out = A.shape[0]
    mud_chain = []

    current_mean = initial_mean
    mud_chain.append(current_mean)
    if idx is None: idx = range(dim_out)
    for i in idx:
        _A = A[i,:].reshape(1,2)
        _b = b[i]
        _y = y[i]
        mud_chain.append(mud_sol(_A, _b, _y, current_mean, initial_cov, data_cov=None))
        current_mean = mud_chain[-1]
    return mud_chain


def iterate(A, b, y, initial_mean, initial_cov, data_cov=None, num_epochs=1, idx=None):
    mud_chain = performEpoch(A, b, y, initial_mean, initial_cov, data_cov, idx)
    for k in range(1, num_epochs):
        mud_chain += performEpoch(A, b, y, mud_chain[-1], initial_cov, data_cov, idx)

    return mud_chain


### OLD CODE ###
### will erase, but committing for posterity ###

# def make_mud_sol(initial_mean, initial_cov, M, data_list, std_of_data, observed_mean=None):
#     data_dimension = len(data_list)  # num QoI
#     if observed_mean is None:
#         observed_mean = np.zeros(data_dimension).reshape(-1,1)
#     if isinstance(std_of_data, float) or isinstance(std_of_data, int):
#         std_of_data = np.ones(len(data_list))*std_of_data
#     # This implements the SWE map.
#     num_obs_per_qoi = [len(dat) for dat in data_list]
#     D = np.diag(np.divide(np.sqrt(num_obs_per_qoi), std_of_data))
#     A = D@M
#     b = -np.array([1./(np.sqrt(num_obs_per_qoi[i])*std_of_data[i])*np.sum(data_list[i]) for i in range(data_dimension)])
#     b = b.reshape(-1,1)
#     y = observed_mean - b - A@initial_mean
#     predicted_cov = A@initial_cov@A.T
#     predicted_prec  = np.linalg.inv(predicted_cov)
#     mud = initial_mean.ravel() + (initial_cov @ A.T @ predicted_prec @ y).ravel()
#     return (A, b, observed_mean, predicted_cov, mud.reshape(-1,1))

# def make_map_sol(prior_mean, prior_cov, data_std, A, data, b): # DO NOTE THAT SIZES HERE ARE Column-Major instead of Row-Major ... (dim, samps)
#     if type(prior_mean) is int:
#         prior_mean = [prior_mean, prior_mean]
#     if type(prior_mean) is float:
#         prior_mean = [prior_mean, prior_mean]
#     if type(prior_mean) is list:
#         prior_mean = np.array(prior_mean).reshape(-1,1)

#     if type(prior_cov) is list:
#         prior_cov = np.diag(prior_cov)

#     if type(data_std) is list:
#         data_std = np.array(data_std).reshape(1,-1)

#     if isinstance(data_std, float) or isinstance(data_std, int):
#         data_cov = data_std*np.eye(len(data))
#     else:
#         assert len(data_std) == len(data), "data and std must match length"

#         data_cov = np.diag(data_std)

#     if type(data) is list:
#         data = np.array(data).reshape(1,-1)
#         print('reformatted data')

# #     prior_mean=initial_mean
# #     prior_cov=initial_cov
# #     data_std=obs_cov
# #     A=A
# #     data=observed_mean



#     precision = np.linalg.inv(A.T@np.linalg.inv(data_cov)@A + np.linalg.inv(prior_cov))
#     kahlman_update = precision@A.T@np.linalg.inv(data_cov)
#     post_mean = prior_mean.ravel() + kahlman_update@(data - b - A@prior_mean).ravel()
#     post_cov = prior_cov - kahlman_update@A@prior_cov

#     return post_mean.reshape(-1,1), post_cov

## mud notebook stuff


if __name__ == "__main__":
    run()

