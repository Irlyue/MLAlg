import numpy as np

from utils import base
from . import svm_solver


KERNELS = {}


def add_kernel(kc):
    KERNELS[kc.NAME] = kc
    return kc


class BaseKernel:
    def __call__(self, X, Y=None):
        """
        :param X: np.array, with shape(n, d)
        :param Y: np.array, with shape(m, d). If None, compute the product value
        of `X` itself.
        :return K: np.array.
            If Y is None, shape(n,)
            If Y is not None, shape(n, m)
        """
        raise NotImplementedError

    def __repr__(self):
        return ('{}({})'.format(type(self).__name__, ', '.join('{}={}'.format(key, value)
                for key, value in vars(self).items())))


@add_kernel
class LinearKernel(BaseKernel):
    NAME = 'linear'

    def __call__(self, X, Y=None):
        return np.sum(X*X, axis=1) if Y is None else np.dot(X, Y.T)


@add_kernel
class PolyKernel(BaseKernel):
    NAME = 'poly'

    def __init__(self, c=1, d=2):
        """
        K(x, y) = (<x, y> + c)^d

        Parameters
        ----------
        :param c: float, defined above
        :param d: float, defined above
        """
        self.c = c
        self.d = d

    def __call__(self, X, Y=None):
        return (np.sum(X*X, axis=1) + self.c)**self.d if Y is None else (np.dot(X, Y.T) + self.c)**self.d


@add_kernel
class RBFKernel(BaseKernel):
    NAME = 'rbf'

    def __init__(self, sigma=1):
        """
        K(x, y) = exp{-||x-y||^2 / (2*sigma^2)}

        Parameters
        ----------
        :param sigma: float, defined above
        """
        self.sigma = sigma

    def __call__(self, X, Y=None):
        if Y is None:
            d = np.zeros(X.shape[0])
        else:
            d = (np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1, keepdims=True).T -
                 2 * np.dot(X, Y.T))
        return np.exp(-d / (2*self.sigma**2))


class SVM(base.BaseModel):
    """
    Attributes
    ----------
    sv: array-like, shape = [k]
        The indices of support vectors.

    A: array-like, shape=[k]
        Non-zero multipliers.

    Ay: array-like, shape = [k]
        The product of Lagrange multiplier alpha and label y. Only those corresponding to
        support vectors are considered.

    b: scalar
        The model parameter b

    w: array-like, shape=[n]
        The model parameter w, only avaliable for linear kernel.

    solver: callable
        The SVM solver.
    """
    def __init__(self, C=1, tau=1e-12, eps=1e-3, kernel='linear', solver=3):
        """
        Parameters
        ----------
        :param C: float, normalization strength.
        :param eps: float, a scalar very closed to zero. Use to verified if the KKT
        conditions hold.
        :param kernel: str or callable, default to linear kernel. Use callable to
        pass in kernel with extra parameters.
        :param solver: int, default to WSS3Solver. Choose from {1, 3}.
        """
        super().__init__()
        self.C = C
        self.tau = tau
        self.eps = eps
        self.kernel = KERNELS[kernel]() if type(kernel) == str else kernel
        solver_cls = getattr(svm_solver, 'WSS%dSolver' % solver)
        self.solver = solver_cls(C=self.C, kernel=self.kernel,
                                 eps=self.eps, tau=self.tau)
        self.freezed = vars(self).copy()

    def fit(self, X, y, seed=None):
        """
        Parameters
        ----------
        :param X: np.array, with shape(m, n)
        :param y: np.array, with shape(m,)

        :return self:
        """
        self.sv, self.A, self.b = self.solver(X, y, seed)
        self.X, self.y = X, y
        self.Ay = self.A * y[self.sv]
        self.on_end_fit()

    def on_end_fit(self):
        pass

    def predict(self, X):
        score = np.sum(self.kernel(X, self.X[self.sv]) * self.Ay, axis=1) + self.b
        y = (score > 0).astype(np.int32)
        y = y * 2 - 1
        return y.flatten()

    def __repr__(self):
        return ('{}({})'.format(type(self).__name__, ', '.join('{}={}'.format(key, value)
                for key, value in self.freezed.items())))


class SVMLinear(SVM):
    """
    Almost the same implementation with the kernel version except that we could obtain
    the actual parameters `w` and `b` when using linear kernel.
    """
    def __init__(self, C=1, tau=1e-12, eps=1e-3, **kwargs):
        super().__init__(C=C, tau=tau, eps=eps, kernel='linear', **kwargs)

    def on_end_fit(self):
        X = self.X
        sv = self.sv
        Ay = self.Ay
        w = np.sum(X[sv] * Ay[:, None], axis=0).reshape((-1, 1))
        self.w = w

    def predict(self, X):
        score = np.dot(X, self.w) + self.b
        y = (score > 0).astype(np.int32)
        y = y * 2 - 1
        return y.flatten()
