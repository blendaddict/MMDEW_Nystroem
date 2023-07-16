import numpy as np
from sklearn import metrics
import numpy.linalg as la
import math
class MMD:
    def __init__(self, biased, gamma=1):
        """ """
        self.biased = biased
        self.gamma = gamma

    def mmd(self, X, Y):
        """Maximum Mean Discrepancy of X and Y."""
        if self.biased:
            XX = metrics.pairwise.rbf_kernel(X, X, self.gamma)
            YY = metrics.pairwise.rbf_kernel(Y, Y, self.gamma)
            XY = metrics.pairwise.rbf_kernel(X, Y, self.gamma)
            return XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            m = len(X)
            n = len(Y)
            XX = metrics.pairwise.rbf_kernel(X, X, self.gamma) - np.identity(m)
            YY = metrics.pairwise.rbf_kernel(Y, Y, self.gamma) - np.identity(n)
            XY = metrics.pairwise.rbf_kernel(X, Y, self.gamma)
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
        X_mn = metrics.pairwise.rbf_kernel(X_tilde, elements)
        XX = metrics.pairwise.rbf_kernel(X_tilde, X_tilde)
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

        XX = metrics.pairwise.rbf_kernel(X_tilde, X_tilde)
        YY = metrics.pairwise.rbf_kernel(Y_tilde, Y_tilde)
        XY = metrics.pairwise.rbf_kernel(X_tilde, Y_tilde)

        X_mn = metrics.pairwise.rbf_kernel(X_tilde, X)
        Y_mn = metrics.pairwise.rbf_kernel(Y_tilde, Y)

        alpha_1 = self.get_alpha(self, XX, X_mn, n)
        alpha_2 = self.get_alpha(self, YY, Y_mn, n)
        return (alpha_1.T @ XX @ alpha_1 + alpha_2.T @ YY @ alpha_2 - 2 * alpha_1.T @ XY @ alpha_2)[0][0]


    def threshold(self, m, n, alpha=0.1):
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
    def estimate_gamma(X, n_samples=200, seed=1234):
        """Estimate the gamma parameter based on the median heuristic for sigma**2."""
        rng = np.random.default_rng(seed)
        distances = []
        for _ in range(n_samples):
            distances += [np.linalg.norm(rng.choice(X, size=2)) ** 2]
            sigma = np.median(distances)
        return 1 / np.sqrt(2 * sigma) if sigma > 0 else 1
