import numpy as np

from utils import base


class SVMLinear(base.BaseModel):
    """
    All the member functions start with '_' do not change the inner state of the class, they
    just calculate the expected values and return them.
    """
    def __init__(self, C=1):
        super().__init__()
        self.C = C

    def fit(self, X, y, eps=1e-6, tol=100, seed=None):
        """
        :param X: np.array, with shape(m, n)
        :param y: np.array, with shape(m,)
        :param eps: float, a float number closed to zero. If |alpha_old - alpha_new| < eps,
        then it's considered as not moving at all, the tolerance counter will increment 
        itself by one.
        :param tol: int, the tolerance counter. If after `tol` loops, we still can't find
        two alpha's to make valid update, then we break out of the loop.
        :param seed: int, default to None. If set, the result would be the same with the
        same seed.
        """
        if seed:
            np.random.seed(seed)
        m, n = X.shape
        counter = 0
        n_loops = 0

        self.X, self.y = X, y
        self.K = np.dot(self.X, self.X.T)
        self.alphas = np.zeros(m)

        while counter < tol:
            counter += 1
            n_loops += 1
            i, j = self._choose_two_alphas()
            L, H = self._calc_LH(i, j)
            if L == H:
                continue

            alphai, alphaj = self._calc_alpha_ij(L, H, i, j)
            if abs(alphai - self.alphas[i]) > eps:
                counter = 0
                self.alphas[i], self.alphas[j] = alphai, alphaj

        self.w, self.b = self._calc_wb()
        print('Done in %d loops.' % n_loops)

    def _choose_two_alphas(self):
        """
        Randomly choose two alpha's.

        :return i, j: the indices of the two alpha's.
        """
        while True:
            i, j = np.random.randint(self.alphas.size, size=(2,))
            if i != j:
                return i, j

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

    def _calc_wb(self):
        """
        After the training, we calculate the original model parameter w and b from
        the Lagrange multipliers.
        """
        alphay = self.alphas * self.y
        w = np.sum(self.X * alphay.reshape((-1, 1)), axis=0).reshape((-1, 1))
        yp, yn = (self.y == 1), (self.y == -1)
        b = -0.5 * (np.max(self.X[yn, :].dot(w)) + np.min(self.X[yp, :].dot(w)))
        return w, b

    def _calc_E(self, i):
        return np.sum(self.K[i, :] * self.alphas * self.y) - self.y[i]

    def _calc_alpha_ij(self, L, H, i, j):
        # calculate alpha_j
        Ei, Ej = self._calc_E(i), self._calc_E(j)
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
