import numpy as np


ALPHA_EPS = 1e-5


class QMatrix:
    """
    This data structure allows access to the matrix element:
        Q[i, j] = y[i]*y[j]*K[i, j], where K is the kernel matrix
    1. Accessing one element, Q[i, j];
    2. Accessing one line, Q[i];
    3. Accessing self product, Q.selfK.

    If the number of training examples is small than 10,000, we just
    pre-compute the whole matrix to avoid repeated calculation.
    """
    def __init__(self, X, y, kernel):
        self.X = X
        self.y = y
        self.kf = kernel
        self.precomute = len(X) < 10000
        if self.precomute:
            self.Q = y[:, None] * self.kf(X, X) * y[None]
        self.selfK = self.kf(X)

    def __getitem__(self, index):
        X, y = self.X, self.y
        if self.precomute:
            return self.Q[index]
        try:
            i, j = index
            return y[i] * y[j] * self.kf(X[i][None], X[j][None])[0, 0]
        except TypeError:
            i = index
            return y[i] * y * self.kf(X[i][None], X)[0]


class Solver:
    def __init__(self, kernel, C=1, eps=1e-3, tau=1e-12, **kwargs):
        self.C = C
        self.kf = kernel
        self.eps = eps
        self.tau = tau
        self.freezed = vars(self).copy()

    def __repr__(self):
        return ('{}({})'.format(type(self).__name__, ', '.join('{}={}'.format(key, value)
                for key, value in self.freezed.items())))

    def __call__(self, X, y):
        raise NotImplementedError

    def calc_LH(self, i, j):
        """
        Calculate the lower and upper bound of alpha[j].
        """
        C = self.C
        y = self.y
        A = self.A
        if y[i] * y[j] == 1:
            L = max(0, A[i] + A[j] - C)
            H = min(C, A[i] + A[j])
        else:
            L = max(0, A[j] - A[i])
            H = min(C, C + A[j] - A[i])
        return L, H

    def on_end(self):
        """
        After obtaining the optimized values of alpha's, we need to compute
        the original model parameter b.

        :return
            sv: the indices of support vectors
            A: the non-zero alpha's
            b: the original model paramter
        """
        X, y = self.X, self.y
        A = self.A
        kf = self.kf
        sv = np.nonzero(A)[0]
        Ay = A[sv] * y[sv]
        yp, yn = (y == 1), (y == -1)
        wXp = np.sum(kf(X[yp], X[sv]) * Ay, axis=1)
        wXn = np.sum(kf(X[yn], X[sv]) * Ay, axis=1)
        b = -0.5 * (wXp.min() + wXn.max())
        return sv, A[sv], b


class WSS1Solver(Solver):
    """
    This solver uses no heuristics to choose alphas, so the runtime may suffer
    for large datasets. Basically, it loops through the whole dataset to look
    for each multiplier that vilates the KKT condition and randomly choose
    another multiplier to perform optimization. If after tol=20 passes through
    the dataset we still cannot find two alpha's to make valid update, the
    algorithm terminates.

    You can find more references in the book, Machine Learning in Action, chapter 6.
    """
    def __init__(self, kernel, tol=-1, **kwargs):
        super().__init__(kernel, **kwargs)
        self.tol = tol if tol != -1 else 20
        self.freezed = {key: value for key, value in vars(self).items() if key != 'freezed'}

    def __call__(self, X, y, seed=None):
        if seed is not None:
            np.random.seed(seed)
        m, n = X.shape
        counter = 0
        n_loops = 0

        self.X, self.y = X, y
        self.K = self.kf(X, X)
        self.b = 0
        A = self.A = np.zeros(m)

        while counter < self.tol:
            counter += 1
            n_loops += 1
            for i in range(m):
                Ei = self.calc_E(i)
                if (y[i]*Ei < -self.eps and A[i] < self.C) or (y[i]*Ei > self.eps and A[i] > 0):
                    j = self.choose_alpha_j(i)
                    Ej = self.calc_E(j)
                else:
                    continue

                Ai, Aj = self.calc_alpha_ij(i, j, Ei, Ej)
                if abs(Ai - A[i]) > ALPHA_EPS:
                    # If any of the alphas are changed, reset the counter
                    counter = 0
                    # update b
                    self.b = self.calc_b(i, j, Ei, Ej, Ai, Aj)
                    A[i], A[j] = Ai, Aj

        print('Done in %d loops.' % n_loops)
        sv, A, b = self.on_end()
        return sv, A, b

    def calc_b(self, i, j, Ei, Ej, Ai, Aj):
        A = self.A
        K = self.K
        y = self.y
        di, dj = A[i] - Ai, A[j] - Aj
        bi = self.b - Ei - y[i]*di*K[i, i] - y[j]*dj*K[i, j]
        bj = self.b - Ej - y[i]*di*K[i, j] - y[j]*dj*K[j, j]
        if 0 < Ai < self.C:
            b = bi
        elif 0 < Aj < self.C:
            b = bj
        else:
            b = (bi + bj) / 2.
        return b

    def choose_alpha_j(self, i):
        """
        Randomly choose the second alpha.

        :return j: the indice of the second alpha.
        """
        while True:
            j = np.random.randint(self.A.size)
            if i != j:
                return j

    def calc_E(self, i):
        return np.sum(self.K[i, :] * self.A * self.y) + self.b - self.y[i]

    def calc_alpha_ij(self, i, j, Ei, Ej):
        """
        Compute the new alpha[i] and alpha[j].
        """
        K = self.K
        A = self.A
        y = self.y
        L, H = self.calc_LH(i, j)
        # calculate alpha_j
        eta = K[i, i] + K[j, j] - 2*K[i, j]
        # only happen when there're duplicate points since
        # <x, x> + <y, y> - 2<x, y> = <x-y, x-y> > 0 if x != y
        if eta <= 0:
            return A[i], A[j]
        Aj = A[j] + y[j]*(Ei - Ej) / eta
        # clip it between [L, H]
        Aj = min(max(L, Aj), H)
        xi = A[i] * y[i] + A[j] * y[j]
        # calculate alpha_i
        Ai = y[i]*(xi - Aj*y[j])
        return Ai, Aj


class WSS3Solver(Solver):
    """
    Introduced in the following paper:
        "Working Set Selection Using Second Order Information for Training Support
    Vector Machines"
    """
    def __call__(self, X, y, seed=None):
        m, n = X.shape
        self.X = X
        self.y = y
        A = self.A = np.zeros(m)
        G = self.G = np.zeros(m) - 1
        Q = self.Q = QMatrix(X, y, self.kf)

        n_loops = 0
        while True:
            n_loops += 1
            i, j = self.select_ij_fast()
            if j == -1:
                break

            a = Q[i, i]+Q[j, j]-2*y[i]*y[j]*Q[i, j]
            a = self.tau if a <= 0 else a
            b = -y[i]*G[i]+y[j]*G[j]

            L, H = self.calc_LH(i, j)
            newAj = A[j] - y[j]*b/a
            newAj = min(max(L, newAj), H)

            xi = A[i]*y[i] + A[j]*y[j]
            newAi = y[i]*(xi - newAj*y[j])
            deltaAi = newAi - A[i]
            deltaAj = newAj - A[j]
            A[i], A[j] = newAi, newAj

            # update gradient
            G += Q[i] * deltaAi + Q[j] * deltaAj
        print('Done in %d loops.' % n_loops)
        sv, A, b = self.on_end()
        return sv, A, b

    def select_ij_slow(self):
        X, y = self.X, self.y
        A, G, Q = self.A, self.G, self.Q
        C = self.C
        Gmin, Gmax = np.Inf, -np.Inf

        i = -1
        for t in range(X.shape[0]):
            if (y[t] == 1 and A[t] < C) or (y[t] == -1 and A[t] > 0):
                Gnow = -y[t] * G[t]
                i, Gmax = (t, Gnow) if Gnow >= Gmax else (i, Gmax)

        j = -1
        obj_min = np.Inf
        for t in range(X.shape[0]):
            if (y[t] == 1 and A[t] > 0) or (y[t] == -1 and A[t] < C):
                Gnow = -y[t]*G[t]
                Gmin = Gnow if Gnow <= Gmin else Gmin
                b = Gmax + y[t]*G[t]
                if b > 0:
                    a = Q[i, i]+Q[t, t]-2*y[i]*y[t]*Q[i, t]
                    a = self.tau if a <= 0 else a
                    obj_now = -(b*b)/a
                    j, obj_min = (t, obj_now) if obj_now <= obj_min else (j, obj_min)

        if Gmax - Gmin < self.eps:
            return -1, -1
        return i, j

    def select_ij_fast(self):
        """
        A vectorized implementation of `select_ij_slow`.
        """
        y = self.y
        A, G, Q = self.A, self.G, self.Q
        C = self.C

        # select i
        flag = ((y == 1) & (A < C)) | ((y == -1) & (A > 0))
        Gs = -y * G
        Gs[~flag] = -np.Inf
        i = Gs.argmax()
        i, Gmax = (i, Gs[i]) if Gs[i] != -np.Inf else (-1, -np.Inf)

        # select j
        flag = ((y == 1) & (A > 0)) | ((y == -1) & (A < C))
        Gs = -y * G
        Gs[~flag] = np.Inf
        Gmin = Gs.min()
        b = Gmax + y * G
        a = Q[i, i] + Q.selfK - 2*y[i]*y*Q[i]
        a[a <= 0] = self.tau
        objs = -(b*b)/a
        flag &= (b > 0)
        objs[~flag] = np.Inf
        j = objs.argmin()

        if Gmax - Gmin < self.eps:
            return -1, -1
        return i, j
