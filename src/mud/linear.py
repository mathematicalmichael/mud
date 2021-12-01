import pdb
import numpy as np
from mud.util import null_space, make_2d_unit_mesh, updated_cov
from numpy.linalg import inv
from matplotlib import cm
from matplotlib import pyplot as plt


class LinearGaussianProblem(object):
    """Inverse Problems with Linear/Affine Maps with Gaussian distributions.

    """

    def __init__(self,
            A=np.array([1, 1]).reshape(-1,1),
            b=None,
            y=None,
            mean_i =None,
            cov_i=None,
            cov_o=None,
            alpha=1.0):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)
        ns, di = self.A.shape

        # Initialize to defaults - Reshape everything into 2D arrays.
        self.b = np.zeros((ns, 1)) if b is None else b.reshape(-1, 1)
        self.y = np.zeros((ns, 1)) if y is None else y.reshape(-1, 1)
        self.mean_i = np.zeros((di, 1)) if mean_i is None else mean_i.reshape(-1, 1)
        self.cov_i = np.eye(di) if cov_i is None else cov_i
        self.cov_o = np.eye(ns) if cov_o is None else cov_o

        # How much to scale regularization terms
        self.alpha = alpha

        # Check appropriate dimensions of inputs
        n_data, n_targets = self.y.shape
        if ns != n_data:
            raise ValueError(
                "Number of samples in X and y does not correspond:"
                " %d != %d" % (ns , n_data)
            )

        # Initialize to no solution
        self.sol = None


    def compute_functionals(self, X, terms='all'):
        """
        For a given input and observed data, compute functionals or
        individual terms in functionals that are minimized to solve the
        linear gaussian problem.
        """
        # Compute observed mean
        mean_o = self.y - self.b

        # Define inner-producted induced by vector norm
        ip = lambda X, mat : np.sum(X * (np.linalg.inv(mat) @ X), axis=0)

        # First compute data mismatch norm
        data_term = ip((self.A @ X.T + self.b) - mean_o.T,
                       self.cov_o)
        if terms=='data': return data_term

        # Tikhonov Regularization Term
        reg_term =  self.alpha * ip((X- self.mean_i.T).T, self.cov_i)
        if terms=='reg': return reg_term

        # Data-Consistent Term - "unregularizaiton" in data-informed directions
        dc_term = self.alpha * ip(self.A @ (X - self.mean_i.T).T,
                             self.A @ self.cov_i @ self.A.T)
        if terms=='dc_term': return dc_term

        # Modified Regularization Term
        reg_m_terms = reg_term - dc_term
        if terms=='reg_m': return reg_m_terms

        bayes_fun = data_term + reg_term
        if terms=='bayes': return bayes_fun

        dc_fun  = bayes_fun - dc_term
        if terms=='dc': return dc_fun

        return (data_term, reg_term, dc_term, bayes_fun, dc_fun)


    def solve(self, method='mud', output_dim=None):
        """
        Explicitly solve linear problem using given method.

        """
        # Reduce output dimension if desired
        od = self.A.shape[0] if output_dim is None else output_dim
        _A = self.A[:od, :]
        _b = self.b[:od, :]
        _y = self.y[:od, :]
        _cov_o = self.cov_o[:od, :od]

        # Compute observed mean
        mean_o = _y - _b

        # Compute residual
        z = mean_o - _A @ self.mean_i

        # Weight initial covariance to use according to alpha parameter
        a_cov_i = self.alpha * self.cov_i

        # Solve according to given method, or solve all methods
        if method == 'mud' or method == 'all':
            inv_pred_cov = np.linalg.pinv(_A @ a_cov_i @ _A.T)
            update = a_cov_i @ _A.T @ inv_pred_cov
            self.mud = self.mean_i + update @ z

        # if method == 'mud_alt' or method == 'all':
        #     up_cov = updated_cov(X=_A, init_cov=a_cov_i, data_cov=_cov_o)
        #     update = up_cov @ _A.T @ np.linalg.inv(_cov_o)
        #     self.mud_alt = self.mean_i + update @ z

        if method == 'map' or method == 'all':
            co_inv = inv(_cov_o)
            cov_p = inv(_A.T @ co_inv @ _A + inv(a_cov_i))
            update = cov_p @ _A.T @ co_inv
            self.map = self.mean_i + update @ z

        if method == 'ls' or method == 'all':
            # Compute ls solution from pinv method
            self.ls = (np.linalg.pinv(_A) @ mean_o)

        # Return solution or all solutions
        if method =='all':
            return (self.mud, self.map, self.ls)
            # return (self.mud, self.mud_alt, self.map, self.ls)
        else:
            return self.__getattribute__(method)


    def plot_sol(self,
            point='mud',
            ax=None,
            label=None,
            note_loc=None,
            pt_opts = {'color':'k', 's':100, 'marker':'o'},
            ln_opts = {'color':'xkcd:blue', 'marker':'d', 'lw':1, 'zorder':10},
            annotate_opts={'fontsize':14, 'backgroundcolor':'w'}):
        """
        Plot solution points
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Get solution point or initial poitn to plot.
        pt = self.mean_i if point=='initial' else self.solve(method=point)
        pt_opts['label'] = point

        # Plot point
        ax.scatter(pt[0], pt[1], **pt_opts)

        # Plot line connecting iniital value and solution
        if ln_opts is not None and point!='initial':
            ax.plot([self.mean_i.ravel()[0], pt.ravel()[0]],
                    [self.mean_i.ravel()[1], pt.ravel()[1]],
                    **ln_opts)

        if label is not None:
            # Annotate point with a label if desired
            nc = note_loc
            nc = (pt[0] - 0.02, pt[1] + 0.02) if nc is None else nc
            ax.annotate(label, nc, **annotate_opts)


    def plot_contours(self,
            ref=None,
            subset=None,
            ax=None,
            annotate=False,
            note_loc = None,
            w=1,
            label = "{i}",
            plot_opts={'color':"k", 'ls':":", 'lw':1, 'fs':20},
            annotate_opts={'fontsize':20}):
        """
        Plot Linear Map Solution Contours
        """
        # Initialize a plot if one hasn't been already
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # All rows of A are default subset of contours to plot
        subset = np.arange(self.A.shape[0]) if subset is None else subset

        # Ref is the reference point to plot each contour line through.
        ref = ref if ref is not None else self.solve(method='ls')

        # Build null-space (contour lines) for each subset row of A
        A = self.A[np.array(subset), :]
        numQoI = A.shape[0]
        AA = np.hstack([null_space(A[i, :].reshape(1, -1)) for i in range(numQoI)]).T

        # Plot each contour line going through ref point
        for i, contour in enumerate(subset):
            xloc = [ref[0] - w * AA[i, 0], ref[1] + w * AA[i, 0]]
            yloc = [ref[0] - w * AA[i, 1], ref[1] + w * AA[i, 1]]
            ax.plot(xloc, yloc, **plot_opts)

            # If annotate is set, then label line with given annotations
            if annotate:
                nl = (xloc[0], yloc[0]) if note_loc is None else note_loc
                ax.annotate(label.format(i=contour + 1), nl, **annotate_opts)


    def plot_fun_contours(self, mesh=None,
            terms='dc', ax=None, N=250, r=1, **kwargs):
        """
        Plot contour map offunctionals being minimized over input space
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Get mesh if one hasn't been passed
        if mesh is None:
            _, _, mesh = make_2d_unit_mesh(N, r)

        # Compute functional terms desired over range
        term = self.compute_functionals(mesh, terms=terms)

        # Plot contours
        _ = ax.contour(mesh[:, 0].reshape(N, N),
                       mesh[:, 1].reshape(N, N),
                       term.reshape(N, N), **kwargs)

class LinearGaussianProblemFromData(LinearGaussianProblem):
    """Inverse Problems with Linear/Affine Maps with Gaussian distributions.

    """

    # TODO: Implement using transform_from_linear_measurements()
    def __init__(self,
            A=np.array([1, 1]).reshape(-1,1),
            b=None,
            y=None,
            mean_i =None,
            cov_i=None,
            cov_o=None,
            alpha=1.0):
        super().__init__(A=A, b=b, y=y, mean_i=mean_i,
                cov_i=cov_i, cov_o=cov_o, alpha=alpha)


class IterativeLinearProblem(LinearGaussianProblem):


    def __init__(self,
            A,
            b,
            y=None,
            initial_mean=None,
            cov=None,
            data_cov=None,
            idx_order=None):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)

        # Initialize to defaults - Reshape everythin into 2D arrays.
        n_samples, dim_input = self.A.shape
        self.data_cov = np.eye(n_samples) if data_cov is None else data_cov
        self.cov = np.eye(dim_input) if cov is None else cov
        self.initial_mean = np.zeros((dim_input, 1)) if initial_mean is None else initial_mean.reshape(-1,1)
        self.b = np.zeros((n_samples, 1)) if b is None else b.reshape(-1,1)
        self.y = np.zeros(n_samples) if y is None else y.reshape(-1,1)
        self.idx_order = range(self.A.shape[0]) if idx_order is None else idx_order

        # Verify arguments?

        # Initialize chain to initial mean
        self.epochs = []
        self.solution_chains = []
        self.errors = []


    def solve(self, num_epochs=1, method='mud'):
        """
        Iterative Solutions
        Performs num_epochs iterations of estimates

        """
        m_init = self.initial_mean if len(self.solution_chains)==0 else self.solution_chains[-1][-1]
        solutions = [m_init]
        for _ in range(0, num_epochs):
            epoch = []
            solutions = [solutions[-1]]
            for i in self.idx_order:
                # Add next sub-problem to chain
                epoch.append(LinearGaussianProblem(self.A[i, :],
                    self.b[i],
                    self.y[i],
                    mean=solutions[-1],
                    cov=self.cov,
                    data_cov=self.data_cov))

                # Solve next mud problem
                solutions.append(epoch[-1].solve(method=method))

            self.epochs.append(epoch)
            self.solution_chains.append(solutions)

        return self.solution_chains[-1][-1]


    def get_errors(self, ref_param):
        """
        Get errors with resepct to a reference parameter

        """
        solutions = np.concatenate([x[1:] for x in self.solution_chains])
        if len(solutions)!=len(self.errors):
            self.errors = [np.linalg.norm(s - ref_param) for s in solutions]
        return self.errors


    def plot_chain(self, ref_param, ax=None, color="k", s=100, **kwargs):
        """
        Plot chain of solutions and contours
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for e, chain in enumerate(self.solution_chains):
            num_steps = len(chain)
            current_point = chain[0]
            ax.scatter(current_point[0], current_point[1], c="b", s=s)
            for i in range(0, num_steps):
                next_point = chain[i]
                points = np.hstack([current_point, next_point])
                ax.plot(points[0, :], points[1, :], c=color)
                current_point = next_point
            ax.scatter(current_point[0], current_point[1], c="g", s=s)
            ax.scatter(ref_param[0], ref_param[1], c="r", s=s)
        self.plot_contours(ref_param, ax=ax, subset=self.idx_order,
                color=color, s=s, **kwargs)


    def plot_chain_error(self, ref_param, ax=None, alpha=1.0,
            color="k", label=None, s=100, fontsize=12):
        """
        Plot error over iterations
        """
        _ = self.get_errors(ref_param)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.set_yscale('log')
        ax.plot(self.errors, color=color, alpha=alpha, label=label)
        ax.set_ylabel("$||\lambda - \lambda^\dagger||$", fontsize=fontsize)
        ax.set_xlabel("Iteration step", fontsize=fontsize)



