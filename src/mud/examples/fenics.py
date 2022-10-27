"""
Fenics Solver

Functions for solving poisson example PDE system of equations using finite
element method using FEniCS. The functions here implement the "forward solver"
for the inverse problems involving the Poisson problem, where the uncertain
parameter is the boundary condition parametrized by a 2-dimensional spline.
These functions can be used to produce new datasets for building inverse
problem solutions using the mud.base.SpatioTemporalProblem class.
"""
import logging
import pickle

import numpy as np
from tqdm import tqdm  # type: ignore

_logger = logging.getLogger(__name__)

fin_flag = False
try:
    import dolfin as fin  # type: ignore

    fin.set_log_level(50)
    fin_flag = True
    fin_reason = None
except Exception as e:
    fin_reason = e


def piecewise_eval_from_vector(u, d=1):
    """
    Takes an iterable `u` with y-values (on interior of equally partitioned unit domain)
    and returns the string for an expression
    based on evaluating a piecewise-linear approximation through these points.
    """
    n = len(u)
    dx = 1 / (n + 1)
    xvals = [i * dx for i in range(n + 2)]
    yvals = [0] + list(u) + [1]

    s = ""
    for i in range(1, len(xvals)):
        start = xvals[i - 1]
        end = xvals[i]
        diff = start - end
        s += f" ((x[{d}] >= {start}) && (x[{d}] < {end}))*"
        s += f"({yvals[i-1]}*((x[{d}]-{end})/{diff}) + "
        s += f"(1 - ((x[{d}]-{end})/{diff}))*{yvals[i]} ) +"

    return s[1:-1]


def fenics_poisson_solve(gamma=-3, lam=None):
    """
    Solve Poisson PDE problem
    """
    # Define mesh
    mesh = fin.UnitSquareMesh(36, 36)

    # Initialize mesh function for interior domains
    domains = fin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)

    # Initialize mesh function for boundary domains
    boundaries = fin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class Left(fin.SubDomain):
        def inside(self, x, on_boundary):
            return fin.near(x[0], 0.0)

    class Right(fin.SubDomain):
        def inside(self, x, on_boundary):
            return fin.near(x[0], 1.0)

    class Bottom(fin.SubDomain):
        def inside(self, x, on_boundary):
            return fin.near(x[1], 0.0)

    class Top(fin.SubDomain):
        def inside(self, x, on_boundary):
            return fin.near(x[1], 1.0)

    # Initialize sub-domain instances
    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom.mark(boundaries, 4)

    if lam is None:
        cons = gamma * 823543 / 12500
        g_L = fin.Expression(f"pow(x[1], 2) * pow(1 - x[1], 5) * {cons}", degree=3)
    else:
        g_L = fin.Expression(piecewise_eval_from_vector(lam), degree=2)

    g_R = fin.Constant(0.0)
    f = fin.Expression(
        "10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2
    )

    # Define function space and basis functions
    V = fin.FunctionSpace(mesh, "Lagrange", 1)
    u = fin.TrialFunction(V)
    v = fin.TestFunction(V)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def top_bottom_boundary(x):
        return x[1] < fin.DOLFIN_EPS or x[1] > 1.0 - fin.DOLFIN_EPS

    dirichlet_bc = fin.Constant(0.0)  # top and bottom
    bcs = fin.DirichletBC(
        V, dirichlet_bc, top_bottom_boundary
    )  # this is required for correct solutions

    # Define new measures associated with the interior domains and
    # exterior boundaries
    dx = fin.Measure("dx", domain=mesh, subdomain_data=domains)
    ds = fin.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Define variational form
    a0 = fin.Constant(1.0)
    F = (
        fin.inner(a0 * fin.grad(u), fin.grad(v)) * dx(0)
        - g_L * v * ds(1)
        - g_R * v * ds(3)
        - f * v * dx(0)
        - f * v * dx(1)
    )

    # Separate left and right hand sides of equation
    a, L = fin.lhs(F), fin.rhs(F)

    # Solve problem
    u = fin.Function(V)
    fin.solve(a == L, u, bcs)

    return u


def run_fenics(
    num_samples,
    num_sensors,
    mins=[-4, -4],
    maxs=[0, 0],
    sensor_low=[0, 0],
    sensor_high=[1, 1],
    gamma=-3,
    save_path=None,
    seed=None,
):
    """
    Run FEniCS to solve a set of Poisson Problems
    """
    if not fin_flag:
        raise ModuleNotFoundError(f"Fenics not found - {fin_reason}")
    if seed is not None:
        np.random.seed(seed)

    if len(mins) != len(maxs):
        raise ValueError("min/max arrays must be of same length")

    sensors = np.random.uniform(low=sensor_low, high=sensor_high, size=(num_sensors, 2))
    lams = np.random.uniform(low=mins, high=maxs, size=(num_samples, len(mins)))
    data = np.zeros((num_samples, num_sensors))
    true_vals = np.zeros((num_sensors, 1))

    for i in tqdm(range(len(lams))):
        u = fenics_poisson_solve(lam=lams[i])
        for j, s in enumerate(sensors):
            data[i, j] = u(s)

    # Solve true solution
    u_true = fenics_poisson_solve(gamma=gamma)
    for j, s in enumerate(sensors):
        true_vals[j] = u_true(s)

    # Store coordinates and grid where true solution solved for plotting later
    c = u_true.function_space().mesh().coordinates()
    v = np.array([u_true(c[i, 0], c[i, 1]) for i in range(len(c))])

    full_res = {
        "lam": lams,
        "data": data,
        "true_vals": true_vals,
        "domain": list(zip(mins, maxs)),
        "sensors": sensors,
        "u": (c, v),
    }

    p = None
    if save_path is not None:
        p = f"{save_path}/s{num_samples}_n{num_sensors}_d{len(mins)}_res"
        with open(p, "wb") as fp:
            pickle.dump(full_res, fp)

    return full_res, p
