import math

import numpy as np
from .mmd import MMD
from sklearn import metrics
import numpy.linalg as la

class BucketStream:
    def __init__(self, gamma, compress=True, alpha=0.1, seed=1234, min_size=200):
        """ """
        self.gamma = gamma
        self.compress = compress
        self.alpha = alpha
        self.buckets = []
        self.maximum_mean_discrepancy = MMD(biased=True, gamma=gamma)
        self.cps = []
        self.rng = np.random.default_rng(seed)
        self.logging=False
        self.min_size = min_size

    def insert(self, element):
        subsample, weights, capacity = self.maximum_mean_discrepancy.get_bucket_content(element)
        self.buckets += [
            Bucket(
                elements=subsample,
                weights=weights,
                capacity=capacity
            )
        ]
        self._find_changes()
        self._merge()

    #ToDo: Delete function
    def mmd(self, split):
        """MMD of the buckets coming before `split` and the buckets coming after `split`, i.e., with 3 buckets and `split = 1` it returns `mmd(b_0, union b1 ... bn)`."""
        start = self.buckets[:split]
        end = self.buckets[split:]

        start_elements = []
        start_weights = []
        end_elements = []
        end_weights = []
        for bucket in start:
            start_elements += bucket.elements
            start_weights += bucket.weights
        for bucket in end:
            end_elements += bucket.elements
            end_weights += bucket.weights
        start_capacity = len(start_elements)
        end_capacity = len(end_elements)
        start_weights = start_weights * (1/start_capacity)
        end_weights = end_weights * (1/end_capacity)
        addend_1 = 0
        addend_2 = 0
        addend_3 = 0
        for i in range(start_capacity):
            for j in range(start_capacity):
                addend_1 += start_weights[i] * start_weights[j] * metrics.pairwise.rbf_kernel(start_elements[i],start_elements[j])
        for i in range(end_capacity):
            for j in range(end_capacity):
                addend_2 += end_weights[i] * end_weights[j] * metrics.pairwise.rbf_kernel(end_elements[i],end_elements[j])
        for i in range(start_capacity):
            for j in range(end_capacity):
                addend_3 += start_weights[i] * end_weights[j] * metrics.pairwise.rbf_kernel(start_elements[i], end_elements[j])
        addend_3 = (-2) * addend_3

        return addend_1 + addend_2 + addend_3

    def _is_change(self, split):
        distance, m, n = self.mmd(split)
        threshold = self.maximum_mean_discrepancy.threshold(m=m, n=n, alpha=self.alpha)
        return distance > threshold

    def _find_changes(self):
        for i in range(1, len(self.buckets)):
            if self._is_change(i):
                if self.compress:
                    position = np.sum([2**len(b.elements) for b in self.buckets[:i]])
                else:
                    position = np.sum([len(b.elements) for b in self.buckets[:i]])
                self.cps = self.cps + [position]
                self.buckets = self.buckets[i:]
                for b in self.buckets:
                    b.XY = b.XY[i:]
                    b.n_XY = b.n_XY[i:]
                return


    def merge_buckets(self, bucket_list):
        """Merges the buckets in `bucket_list` such that one bucket remains with XX, and XY such that their values correspond to the case that all data would have been in this bucket."""
        if len(bucket_list) == 1:
            return bucket_list[0]
        current = bucket_list[-1]
        previous = bucket_list[-2]

        current_elements = current.elements
        previous_elements = previous.elements

        n = len(current_elements)  # assuming current_elements and previous_elements have the same length
        m = round(math.sqrt(2*n))

        m_idx = np.random.default_rng().integers(n, size=m)
        merged_elements = np.concatenate(current_elements,previous_elements)
        subsample = merged_elements[m_idx]
        K_m = np.zeros((m, m))  # initialize the kernel matrix with zeros
        for i in range(m):
            for j in range(m):
                # reshape to 2D array as rbf_kernel expects 2D array
                a = subsample[i].reshape(1, -1)
                b = subsample[j].reshape(1, -1)
                K_m[i, j] = metrics.pairwise.rbf_kernel(a, b)
        K_m_inv = la.pinv(K_m)
        K_mn = np.zeros((n, m))
        for i in range(m):
            for j in range(n):
                a = subsample[i].reshape(1, -1)
                b = merged_elements[j].reshape(1, -1)
                K_mn[i, j] = metrics.pairwise.rbf_kernel(a, b)
        newWeights = (1/n) * K_m_inv @ K_mn @ np.ones((n, 1))

        return self.merge_buckets(
            bucket_list[:-2]
            + [
                Bucket(
                    elements=subsample,
                    weights=newWeights,
                    capacity=m,
                )
            ]
        )


    def get_changepoints(self):
        return np.cumsum(self.cps)

    def _merge(self):
        if len(self.buckets) < 2:
            return
        current = self.buckets[-1]
        previous = self.buckets[-2]
        if previous.capacity == current.capacity:
            self.buckets = self.buckets[:-2] + [self.merge_buckets(self.buckets[-2:])]
            self._merge()

    def k(self, x, y):
        #return np.dot(x,y)
        squared_norm = np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
        return np.exp(-self.gamma * squared_norm)

    def xy(self, element):
        XY = []
        n_XY = []
        for b in self.buckets:
            ret = 0
            n = 0
            for y in b.elements:
                ret += self.k(element, y)
                n += 1
            XY += [ret]
            n_XY += [n]
        return XY, n_XY

    def str_k(self, x, y):
        return f"k({x},{y})"


    def str_xy(self, element):
        XY = []
        for b in self.buckets:
            ret = ""
            for y in b.elements:
                ret += self.str_k(element, y)
            XY += [ret]
        return XY


    def __str__(self):
        return "\n\n".join([str(b) for b in self.buckets])


class Bucket:
    def __init__(self, elements, capacity, weights):
        """ """
        self.elements = elements
        self.capacity = capacity
        self.weights = weights

    def __str__(self):
        return f"Elems:\t{self.elements}\nXX:\t{self.weights}"


if __name__ == "__main__":
    bs = BucketStream(gamma=1, compress=False)
    bs.insert(np.array([1]))
    bs.insert(np.array([2]))
    bs.insert(np.array([3]))
    bs.insert(np.array([4]))
    # bs.insert(5)
    # bs.insert(6)
    # bs.insert(7)
    # bs.insert(8)
    print(bs)
    print(bs.buckets[0].XX)

## Tests
def k(x, y):
    squared_norm = np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
    return np.exp(-1 * squared_norm)


def test_XX():
    gamma = 1
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(200, 2))  # auch mit 2D Daten testen

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    for b in bs.buckets:
        expected = np.sum(metrics.pairwise.rbf_kernel(b.elements, gamma=gamma))
        assert abs(expected - b.XX) < 10e-6


def test_XY():
    gamma = 1
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(2 ** 7 - 1, 2))  # auch mit 2D Daten testen

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    for i in range(1, len(bs.buckets)):
        current = bs.buckets[i]

        for j in range(len(current.XY)):
            expected = np.sum(
                metrics.pairwise.rbf_kernel(
                    bs.buckets[j].elements, current.elements, gamma=gamma
                )
            )
            assert abs(expected - current.XY[j]) < 10e-6


def test_merge_buckets_all_data():
    gamma = 1
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(2 ** 6 - 1, 2))  # auch mit 2D Daten testen

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)
    assert (
        abs(bs.merge_buckets(bs.buckets).XX)
        - np.sum(metrics.pairwise.rbf_kernel(data, gamma=gamma))
        < 10e-6
    )


def test_merge_buckets_with_split_after_first_bucket():
    gamma = 1
    rng = np.random.default_rng(1234)
    size = 7
    data = rng.normal(size=(2 ** size - 1, 2))  # auch mit 2D Daten testen
    X = data[: 2 ** (size - 1)]
    Y = data[2 ** (size - 1) :]

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    x_bucket = bs.buckets[0]
    y_bucket = bs.merge_buckets(bs.buckets[1:])

    XX_expected = np.sum(metrics.pairwise.rbf_kernel(X, gamma=gamma))
    YY_expected = np.sum(metrics.pairwise.rbf_kernel(Y, gamma=gamma))
    XY_expected = np.sum(metrics.pairwise.rbf_kernel(X, Y, gamma=gamma))
    assert abs(XX_expected - x_bucket.XX) < 10e-6
    assert abs(YY_expected - y_bucket.XX) < 10e-6
    assert abs(XY_expected - y_bucket.XY[0]) < 10e-6


def test_merge_buckets_with_split_after_second_bucket():
    gamma = 1
    rng = np.random.default_rng(1234)
    size = 7
    data = rng.normal(size=(2 ** size - 1, 2))  # auch mit 2D Daten testen
    X = data[: 2 ** (size - 1) + 2 ** (size - 2)]
    Y = data[2 ** (size - 1) + 2 ** (size - 2) :]

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    x_bucket = bs.merge_buckets(bs.buckets[:2])
    y_bucket = bs.merge_buckets(bs.buckets[2:])

    XX_expected = np.sum(metrics.pairwise.rbf_kernel(X, gamma=gamma))
    YY_expected = np.sum(metrics.pairwise.rbf_kernel(Y, gamma=gamma))
    XY_expected = np.sum(metrics.pairwise.rbf_kernel(X, Y, gamma=gamma))
    assert abs(XX_expected - x_bucket.XX) < 10e-6
    assert abs(YY_expected - y_bucket.XX) < 10e-6
    assert abs(XY_expected - np.sum(y_bucket.XY)) < 10e-6


def test_mmd():
    split = 2
    gamma = 1
    rng = np.random.default_rng(1234)
    size = 7
    data = rng.normal(size=(2 ** size - 1, 2))  # auch mit 2D Daten testen
    X = data[: 2 ** (size - 1) + 2 ** (size - 2)]
    Y = data[2 ** (size - 1) + 2 ** (size - 2) :]

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    expected = (MMD(biased=True, gamma=gamma)).mmd(X, Y)

    assert abs(expected - bs.mmd(split)[0]) < 10e-6


def test_change_detection():
    gamma = 1
    rng = np.random.default_rng(1234)
    size = 7
    X = rng.normal(size=(2 ** size, 2))
    Y = rng.uniform(size=(2 ** size - 1, 2))
    data = np.vstack((X, Y, X, Y))

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    assert list(bs.get_changepoints()) == [128, 256, 384]


def test_n_XX():
    gamma = 1
    size = 7
    data = np.arange(size).reshape(-1, 1)

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)

    assert [16, 4, 1] == [b.n_XX for b in bs.buckets]


def test_n_XY():
    gamma = 1
    size = 7
    data = np.arange(size).reshape(-1, 1)

    bs = BucketStream(gamma=gamma, compress=False)
    for elem in data:
        bs.insert(elem)
    expected = [[], [8], [4, 2]]
    actual = [list(b.n_XY) for b in bs.buckets]
    assert expected == actual


def test_compression():
    rng = np.random.default_rng(1234)
    size = 8
    X = rng.normal(size=(2 ** size, 10))
    Y = rng.uniform(size=(2 ** size, 10))
    data = np.vstack((X, Y, X))
    # data = X[:44]
    gamma = MMD.estimate_gamma(data)
    bs = BucketStream(gamma=gamma, compress=True, alpha=0.1)
    for elem in data:
        bs.insert(elem)
    assert [256, 512] == list(bs.get_changepoints())
