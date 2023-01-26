import pandas as pd
import numpy as np
from mud import plot as plt
from mud.base import LinearGaussianProblem as LGP
import concurrent.futures
from alive_progress import alive_bar


def generate_a(n, c):
    """
    Random Condition Matrix

    Generates an n-dimensional square matrix A
    with conditiona number c.
    """
    a = np.random.randn(n, n)
    u, s, vh = np.linalg.svd(a)
    s = np.linspace(c, 1, n)
    a = np.dot(u, np.dot(np.diag(s), vh))

    return a


def solve_mud_up_cov(a=None, n=100, c=100):
    """
    Solve MUD Update Covariance

    Solve mud problem using alternate and main method. Compare results and
    condition number of updated covariance
    """
    a = generate_a(n, c)
    lgp = LGP(A=a)
    lgp = LGP(A=a)
    mud = lgp.solve(method="mud")
    mud_alt = lgp.solve(method="mud_alt")
    up_cov = lgp.up_cov
    up_cov_c = np.linalg.cond(up_cov)
    diff = np.linalg.norm(np.abs(mud - mud_alt))

    return diff, up_cov_c


def generate_matrices(max_c):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_matrices = {
            executor.submit(solve_mud_up_cov, 100, c): c
            for c in range(1, max_c + 1, 10)
        }
        with alive_bar(len(future_matrices),
                       unknown="waves",
                       bar="bubbles",
                       spinner="dots_waves",
                       receipt=False) as progress_bar:
            for future in concurrent.futures.as_completed(future_matrices):
                progress_bar()
                results.append(future.result())

    return pd.DataFrame(results)



