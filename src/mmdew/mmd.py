import numpy as np
from sklearn import metrics
import numpy.linalg as la
import math
class MMD:
    def __init__(self, biased, gamma=1, kernel = "rbf"):
        """ """
        self.kernel = kernel
        self.biased = biased
        self.gamma = gamma

    def k(self, x,y,gamma=1):
        if self.kernel == "rbf":
            return metrics.pairwise.rbf_kernel(x, y,gamma)
        else:
            return metrics.pairwise.linear_kernel(x,y)

    def mmd(self, X, Y):
        """Maximum Mean Discrepancy of X and Y."""
        if self.biased:
            XX =  self.k(X, X, self.gamma)
            YY =  self.k(Y, Y, self.gamma)
            XY =  self.k(X, Y, self.gamma)
            return XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            m = len(X)
            n = len(Y)
            XX =  self.k(X, X, self.gamma) - np.identity(m)
            YY =  self.k(Y, Y, self.gamma) - np.identity(n)
            XY =  self.k(X, Y, self.gamma)
            return (
                1 / (m * (m - 1)) * np.sum(XX)
                + 1 / (n * (n - 1)) * np.sum(YY)
                - 2 * XY.mean()
            )

    def get_alpha(self, XX, X_mn, n=1):
        return 1 / n * la.pinv(XX) @ X_mn @ np.ones((n, 1))

    def get_bucket_content(self, element, m=0):
        elements = np.array([element])
        #X = X.reshape(1, -1)
        print("le_X:")
        print(elements)
        print("le_X_shape:")
        print(elements.shape)
        #m_idx = np.random.default_rng().integers(n, size=m)
        X_tilde = elements
        X_mn =  self.k(X_tilde, elements)
        XX = self.k(X_tilde, X_tilde)
        print("Le shapes")
        print(X_mn.shape)
        print(XX.shape)
        alpha = self.get_alpha(XX, X_mn)
        return X_tilde, alpha, m

    def nystroem_mmd(self, X, Y, m):
        """Maximum Mean Discrepancy of approximator X and Y."""
        n = len(X)
        # m = int(m_magnitude(n))
        m_idx = np.random.default_rng().integers(n, size=m)
        X_tilde = X[m_idx]
        Y_tilde = Y[m_idx]

        XX = self.k(X_tilde, X_tilde)
        YY = self.k(Y_tilde, Y_tilde)
        XY =  self.k(X_tilde, Y_tilde)

        X_mn = self.k(X_tilde, X)
        Y_mn = self.k(Y_tilde, Y)
       
        alpha_1 = self.get_alpha( XX, X_mn, n)
 
        alpha_2 = self.get_alpha( YY, Y_mn, n)
        return (alpha_1.T @ XX @ alpha_1 + alpha_2.T @ YY @ alpha_2 - 2 * alpha_1.T @ XY @ alpha_2)[0][0]

    def threshold(self, m, n, alpha):
        K = 1
        if self.biased:
            thresh = (
                np.sqrt(K / m + K / n)
                + np.sqrt((2 * K * (m + n) * np.log(1 / alpha)) / (m * n))
            ) ** 2

            return thresh  # square to be consistent with unbiased threshold
        else:
            raise NotImplementedError

    def threshold_old(self, m, n, alpha=0.1):
        """Using the original corollary from the paper (corollary 9)"""
        K = 1
        m = max(m, n)
        if self.biased:
            return (
                np.sqrt(2 * K / m) * (1 + np.sqrt(2 * np.log(1 / alpha)))
            ) ** 2  # square to be consistent with unbiased threshold
        else:
            return (4 * K / np.sqrt(m)) * np.sqrt(np.log(1 / alpha))

    def accept_H0(self, X, Y, alpha=0.1):
        """Test whether the distributions of X and Y are the same with significance alpha. True if both distributions are the same."""
        return self.mmd(X, Y) < self.threshold(len(X), len(Y), alpha=alpha)

    @staticmethod
    def estimate_gamma(X, max_len=500):
        """Estimate the gamma parameter based on the median heuristic for sigma**2."""

        n = min(len(X), max_len)
        dists = []
        for i in range(n):
            for j in range(i, n):
                dists += [np.linalg.norm(X[i] - X[j], ord=2) ** 2]
        bw = np.median(dists)
        return np.sqrt(bw * 0.5)

