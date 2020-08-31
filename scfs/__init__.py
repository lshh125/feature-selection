import numpy
import scipy
import warnings
from scipy.stats import entropy
from scipy.optimize import minimize
from itertools import chain


def reversed_range(start, end = 0):
    return range(start - 1, end - 1, -1)


class Fspsn:
    """
    Coded up by Shaoheng Liang

    Reference:
    Xiaokai Wei, Philip S. Yu
    Unsupervised feature selection by preserving stochastic neighbors.
    http://proceedings.mlr.press/v51/wei16.pdf
    """

    def __init__(self, l1 = 0.01, eps = 0.01):
        """

        :param l1: l1 regularization strength
        :param eps: epsilon for an entry in w to be projected to (if the gradient also agrees).
        """
        self.l1 = l1
        self.eps = eps

    def fit(self, X, w0 = None, method='L-BFGS-B'):
        """
        Fit FSPSN model on a matrix X.
        :param X: Data matrix. Each row is a observation (cell) and each column is a feature (gene)
        :param w0: initial w, [1, 1, ..., 1] if None
        :return: Fitted w
        """
        n, d = X.shape
        if w0 is None:
            w0 = numpy.ones(d)

        P, beta = self.get_P_and_beta(X)

        if method == 'L-BFGS-B':
            res = minimize(fun=lambda w: self.loss_(P, self.get_Q(X, w, self.beta), w),
                           x0=w0,
                            method='L-BFGS-B',
                            jac=lambda w: self.grad_(P, self.get_Q(X, w, self.beta)),
                            bounds=[(0, 1) for i in range(d)]
                           )
            self.w = res.x
        elif method == 'L-BFGS-Projection':
            raise NotImplemented("")
        else:
            raise ValueError("Available optimization methods are L-BFGS-B and L-BFGS-Projection")

        return self.w

    def lbfgsp(self, fun, x0, jac, free_var_th = 1e-2, m = 5, max_step = 1.0, c = 0.1, delta = 0.5,
               tol = 1e-6, max_iter = 100, max_armijo_iter=10):
        n = x0.shape[0]
        rec = numpy.empty([max_iter + 1, n])
        rec[:] = numpy.nan
        x = x0.copy()

        delta_x = numpy.empty([m, n])
        delta_grad = numpy.empty([m, n])
        rho = numpy.empty(m)

        for iter in range(max_iter):
            iter_mod_m = iter % m

            rec[iter, :] = x
            print(iter + .1, x, fun(x))

            prev_x = x
            x = x.copy()

            # Identify restricted and free variables by Eq(10) and Eq(11).
            grad = numpy.squeeze(jac(x))
            mask0 = numpy.full(m, False)
            mask1 = numpy.full(m, False)
            mask0[(x < free_var_th) & (grad > 0.)] = True
            mask1[(x > 1. - free_var_th) & (grad < 0.)] = True
            mask_free_var = ~(mask0 | mask1)

            # Set the restricted variables to the corresponding lower or upper bound (i.e., 0 or 1)
            x[mask0] = 0.
            x[mask1] = 1.

            fx = fun(x)
            print(iter + .2, x, fx)
            grad = numpy.squeeze(jac(x))

            if iter == 0:
                d = -grad
            else:
                iter_1_mod_m = (iter - 1) % m
                delta_grad[iter_1_mod_m] = grad - prev_grad
                rho[iter_1_mod_m] = 1 / numpy.inner(delta_grad[iter_1_mod_m], delta_x[iter_1_mod_m])

                # two-loop recursion
                range1 = reversed_range(iter_mod_m) if iter < m else chain(reversed_range(iter_mod_m), reversed_range(m, iter_mod_m))
                range2 = range(iter_mod_m) if iter < m else chain(range(iter_mod_m, m), range(iter_mod_m))
                q = grad[mask_free_var]

                n_free_var = numpy.round(mask_free_var.sum())
                alpha = numpy.empty(n_free_var)
                beta = numpy.empty(n_free_var)
                for i in range1:
                    alpha[i] = rho[i] * numpy.inner(delta_x[i, mask_free_var], q)
                    q = q - alpha[i] * delta_grad[i, mask_free_var]
                d = numpy.inner(delta_x[iter_1_mod_m, mask_free_var], delta_grad[iter_1_mod_m, mask_free_var]) /\
                    numpy.inner(delta_grad[iter_1_mod_m, mask_free_var], delta_grad[iter_1_mod_m, mask_free_var]) * q
                #d = q
                for i in range2:
                    beta[i] = rho[i] * numpy.inner(delta_grad[i, mask_free_var], d)
                    d = d + (alpha[i] - beta[i]) * delta_x[i, mask_free_var]
                d = -d

            prev_grad = grad

            # Armijo
            full_d = numpy.zeros(m)
            full_d[mask_free_var] = d
            armijo_success_flag = False
            step = max_step
            for armijo_iter in range(max_armijo_iter):
                if fun(x + step * full_d) <= fx + c * step * (full_d @ grad):
                    armijo_success_flag = True
                    break
                else:
                    step = step * delta

            if not armijo_success_flag:
                warnings.warn("Maximum iteration for Armijo line search is reached.")

            x = x + full_d * step
            delta_x[iter_mod_m] = x - prev_x
            if (delta_x[iter_mod_m, :] ** 2).sum() < tol:
                break

        rec[max_iter, :] = x
        print(max_iter, x, fun(x))

        return rec

    def fit_transform(self):
        pass

    def transform(self, other_X):
        if self.w is None:
            raise RuntimeError("Model has not been fitted!")
        else:
            pass

    def loss_(self, P, Q, w):
        return entropy(P, Q) + self.l1 * numpy.abs(w)

    def grad_(self, P, Q, X, beta):
        n, d = P, Q
        P_Q = P - Q

        res = numpy.zeros(d)
        for t in range(d):
            res[t] = (P_Q * numpy.subtract.outer(X[:, t], X[:, t]) * beta.reshape([-1, 1])).sum() + self.l1

    @staticmethod
    def Hbeta(D=numpy.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """

        # Compute P-row and corresponding perplexity
        P = numpy.exp(-D.copy() * beta)
        sumP = sum(P)
        # print(sumP)
        # H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        H = entropy(P)
        return H, P

    @staticmethod
    def get_Q(X, w, beta):
        (n, d) = X.shape
        X = X * w
        sum_X = numpy.sum(numpy.square(X), 1)
        D = numpy.add(numpy.add(-2 * numpy.dot(X, X.T), sum_X).T, sum_X)
        (H, Q) = Fspsn.Hbeta(D, beta)
        return Q

    @staticmethod
    def get_P_and_beta(X=numpy.array([]), tol=1e-5, perplexity=30.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        print("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = numpy.sum(numpy.square(X), 1)
        D = numpy.add(numpy.add(-2 * numpy.dot(X, X.T), sum_X).T, sum_X)
        P = numpy.zeros((n, n))
        beta = numpy.ones((n, 1))
        logU = numpy.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -numpy.inf
            betamax = numpy.inf
            Di = D[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i + 1:n]))]
            (H, thisP) = Fspsn.Hbeta(Di, beta[i])

            # if i % 500 == 0:
            # print(H, thisP)

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0

            while (not numpy.abs(Hdiff) < tol) and tries < 50:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == numpy.inf or betamax == -numpy.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == numpy.inf or betamin == -numpy.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = Fspsn.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        print("Mean value of sigma: %f" % numpy.mean(numpy.sqrt(1 / beta)))
        return P, beta
