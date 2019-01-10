import numpy as np

from utils import base

ALPHA_EPS = 1e-5
KERNELS = {}


def add_kernel(kc):
    KERNELS[kc.NAME] = kc
    return kc


class BaseKernel:
    def __call__(self, X, Y):
        """
        :param X: np.array, with shape(n, d)
        :param Y: np.array, with shape(m, d)
        :return K: np.array, with shape(n, m)
        """
        raise NotImplementedError

    def __repr__(self):
        return ('{}({})'.format(type(self).__name__, ', '.join('{}={}'.format(key, value)
                for key, value in vars(self).items())))


@add_kernel
class LinearKernel(BaseKernel):
    NAME = 'linear'

    def __call__(self, X, Y):
        return np.dot(X, Y.T)


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

    def __call__(self, X, Y):
        return (np.dot(X, Y.T) + self.c)**self.d


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

    def __call__(self, X, Y):
        d = (np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1, keepdims=True).T -
             2 * np.dot(X, Y.T))
        return np.exp(-d / (2*self.sigma**2))


class SVM(base.BaseModel):
    """
    Attributes
    ----------
    sv: array-like, shape = [n, d]
        Support vectors.

    alphays: array-like, shape = [n]
        The product of Lagrange multiplier alpha and label y. Only those corresponding to
        support vectors are considered.
    """
    def __init__(self, C=1, tol=10, eps=1e-3, kernel='linear'):
        """
        Parameters
        ----------
        :param C: float, normalization strength.
        :param eps: float, a scalar very closed to zero. Use to verified if the KKT
        conditions hold.
        :param tol: int, the tolerance counter. If after `tol` loops, we still can't
        find two alphas to make a valid update, then we break out of the loop.
        :param kernel: str or callable, default to linear kernel. Use callable to
        pass in kernel with extra parameters.
        """
        super().__init__()
        self.C = C
        self.tol = tol
        self.eps = eps
        self.kernel = KERNELS[kernel]() if type(kernel) == str else kernel
        self.freezed = vars(self).copy()

    def fit(self, X, y, seed=None):
        """
        Parameters
        ----------
        :param X: np.array, with shape(m, n)
        :param y: np.array, with shape(m,)
        :param seed: int, default to None. If set, the result would be the same with the
        same seed.
        """
        if seed is not None:
            np.random.seed(seed)
        m, n = X.shape
        counter = 0
        n_loops = 0

        self.X, self.y = X, y
        self.K = self.kernel(self.X, self.X)
        self.alphas = np.zeros(m)
        self.b = 0

        while counter < self.tol:
            counter += 1
            n_loops += 1

            # loop through each multiplier
            changed = False
            for i in range(m):
                if self.inner_loop(i):
                    changed = True
                    counter = 0

            # loop through those non-bound multipliers
            while changed:
                changed = False
                unbound = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in unbound:
                    if self.inner_loop(i):
                        changed = True

        print('Done in %d loops.' % n_loops)
        self.on_end_fit()

    def inner_loop(self, i):
        Ei = self._calc_E(i)
        if (self.y[i] * Ei < -self.eps and self.alphas[i] < self.C) or \
           (self.y[i] * Ei > self.eps and self.alphas[i] > 0):
            # j, Ej = self._select_alpha_j(i, Ei)
            j = self._choose_alpha_j(i)
            Ej = self._calc_E(j)
            alphai, alphaj = self._calc_alpha_ij(i, j, Ei, Ej)
            if abs(alphaj - self.alphas[j]) > ALPHA_EPS:
                # update b
                self.b = self._calc_b(i, j, Ei, Ej, alphai, alphaj)
                self.alphas[i], self.alphas[j] = alphai, alphaj
                return True
        return False

    def _calc_b(self, i, j, Ei, Ej, alphai, alphaj):
        di, dj = alphai - self.alphas[i], alphaj - self.alphas[j]
        bi = self.b - Ei - self.y[i]*di*self.K[i, i]-self.y[j]*dj*self.K[i, j]
        bj = self.b - Ej - self.y[i]*di*self.K[i, j]-self.y[j]*dj*self.K[j, j]
        if 0 < alphai < self.C:
            b = bi
        elif 0 < alphaj < self.C:
            b = bj
        else:
            b = (bi + bj) / 2.
        return b

    def on_end_fit(self):
        """
        Store the support vectors for prediction.
        """
        flag = self.alphas > 0.
        self.alphays = self.alphas[flag] * self.y[flag]
        self.sv = self.X[flag].copy()

    def _choose_alpha_j(self, i):
        """
        Randomly choose the second alpha.

        :return j: the index of the second alpha.
        """
        while True:
            j = np.random.randint(self.alphas.size)
            if i != j:
                return j

    def _calc_LH(self, i, j):
        """
        Calculate the lower and upper bound of alpha[j].
        """
        C = self.C
        if self.y[i] * self.y[j] == 1:
            L = max(0, self.alphas[i] + self.alphas[j] - C)
            H = min(C, self.alphas[i] + self.alphas[j])
        else:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(C, C + self.alphas[j] - self.alphas[i])
        return L, H

    def _calc_E(self, i):
        return np.sum(self.K[i, :] * self.alphas * self.y) + self.b - self.y[i]

    def _calc_alpha_ij(self, i, j, Ei, Ej):
        L, H = self._calc_LH(i, j)
        # calculate alpha_j
        d = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
        # only happen when there're duplicate points since
        # <x, x> + <y, y> - 2<x, y> = <x-y, x-y> > 0 if x != y
        if d == 0:
            return self.alphas[i], self.alphas[j]
        alphaj = self.alphas[j] + self.y[j] * (Ei - Ej) / d
        # clip it between [L, H]
        alphaj = min(max(L, alphaj), H)
        xi = self.alphas[i]*self.y[i] + self.alphas[j]*self.y[j]
        # calculate alpha_i
        alphai = self.y[i]*xi - alphaj*self.y[i]*self.y[j]
        return alphai, alphaj

    def predict(self, X):
        score = np.sum(self.kernel(X, self.sv) * self.alphays[None], axis=1) + self.b
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
    def __init__(self, C=1, tol=10, eps=1e-3):
        super().__init__(C=C, tol=tol, eps=eps, kernel='linear')

    def on_end_fit(self):
        self.w, self.b = self._calc_wb()

    def _calc_wb(self):
        """
        After the training, we calculate the original model parameter w and b from
        the Lagrange multipliers.
        """
        flag = self.alphas > self.eps
        alphay = self.alphas[flag] * self.y[flag]
        w = np.sum(self.X[flag] * alphay[:, None], axis=0).reshape((-1, 1))
        yp, yn = (self.y == 1), (self.y == -1)
        b = -0.5 * (np.max(self.X[yn].dot(w)) + np.min(self.X[yp].dot(w)))
        return w, b

    def predict(self, X):
        score = np.dot(X, self.w) + self.b
        y = (score > 0).astype(np.int32)
        y = y * 2 - 1
        return y.flatten()
