import math

import numpy as np
from .mmd import MMD
from sklearn import metrics
import numpy.linalg as la


class BucketStream:
    def __init__(self, gamma,  alpha=0.1, seed=1234, min_size=200, apply_subsampling=True):
        """ """
        self.apply_subsampling = apply_subsampling
        self.gamma = gamma
        self.alpha = alpha
        self.buckets = []
        self.maximum_mean_discrepancy = MMD(biased=True, gamma=gamma)
        self.cps = []
        self.cps = []
        self.rng = np.random.default_rng(seed)
        self.logging=False
        self.min_size = min_size

        #remove this after testing:
        self.started_ss = False
    def insert(self, element):
        #print("inserting element")
        #print(element.shape)
        #print(np.array([element]).reshape(-1,1).shape)

        self.buckets += [
            Bucket(
                elements=np.array(element).reshape(1,-1),
                weights=np.array([1]).reshape(-1,1),
                capacity=1,
                uncompressed_capacity=1
            )
        ]
        self._find_changes()
        self._merge()

    
    def inv(self, A, l=1e-8):
        
        return np.linalg.inv(A + l * np.identity(len(A))) 

    #only for testing purposes
    def insert_no_cut(self, element):
        #breakpoint()
        self.buckets += [
            Bucket(
                elements=np.array(element).reshape(1,-1),
                weights=np.array(1).reshape(1,-1),
                capacity=1,
                uncompressed_capacity=1
            )
        ]
        #breakpoint()
        self._merge()
        #breakpoint()
        kek = "test"


    def k(self, x, y):

        
        return metrics.pairwise.rbf_kernel(x,y, gamma=self.gamma)
        #return metrics.pairwise.linear_kernel(x,y)
        #squared_norm = np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
        #return np.exp(-self.gamma * squared_norm)

    #ToDo: Delete function
    def mmd(self, split):
        """MMD of the buckets coming before `split` and the buckets coming after `split`, i.e., with 3 buckets and `split = 1` it returns `mmd(b_0, union b1 ... bn)`."""
        start = self.buckets[:split]
        end = self.buckets[split:]
        #breakpoint()
        start_elements = start[0].elements
        start_weights = start[0].weights * start[0].uncompressed_capacity
        end_elements = end[0].elements
        end_weights = end[0].weights * end[0].uncompressed_capacity
        start_uncompressed_capacity = start[0].uncompressed_capacity
        end_uncompressed_capacity = end[0].uncompressed_capacity
        #breakpoint()
        for bucket in start[1:]:
            #breakpoint()
            start_elements = np.concatenate((start_elements, bucket.elements))
            start_weights = np.concatenate((start_weights, bucket.weights * bucket.uncompressed_capacity))
            start_uncompressed_capacity += bucket.uncompressed_capacity
        for bucket in end[1:]:
            # breakpoint()
            end_elements = np.concatenate((end_elements, bucket.elements))
            # breakpoint()
            end_weights = np.concatenate((end_weights, bucket.weights * bucket.uncompressed_capacity))
            end_uncompressed_capacity += bucket.uncompressed_capacity
        #

        start_weights = start_weights * (1 / start_uncompressed_capacity)
        end_weights = end_weights * (1 / end_uncompressed_capacity)

        addend_1 = start_weights.T @ self.k(start_elements, start_elements) @ start_weights
        addend_2 = end_weights.T @ self.k(end_elements, end_elements) @ end_weights
        addend_3 = start_weights.T @ self.k(start_elements, end_elements) @ end_weights

        #print(f"split: {split} start_uncompressed_capacity: {start_uncompressed_capacity} und end_uncompressed_capacity: {end_uncompressed_capacity}")
        return (addend_1 + addend_2 - (2 * addend_3))[0][0], start_uncompressed_capacity, end_uncompressed_capacity


    def _is_change(self, split):
        distance, m, n = self.mmd(split)

        threshold = self.maximum_mean_discrepancy.threshold(m=m, n=n, alpha=self.alpha)
        #print(f"Distance: {distance}, Threshold: {threshold}")
        return distance > threshold

    def _find_changes(self):
        for i in range(1, len(self.buckets)):

            if self._is_change(i):
                position = np.sum([b.uncompressed_capacity for b in self.buckets[:i]])
                self.cps = self.cps + [position]
                self.buckets = self.buckets[i:]
                #Warum return? Will man nicht mehrere CPs finden?
                return




    def merge_buckets(self, bucket_list):
        """Merges the buckets in `bucket_list` such that one bucket remains with XX, and XY such that their values correspond to the case that all data would have been in this bucket."""
        if len(bucket_list) == 1:
            return bucket_list[0]
        current = bucket_list[-1]
        previous = bucket_list[-2]

        current_elements = current.elements
        previous_elements = previous.elements
        current_weights = current.weights
        previous_weights = previous.weights

        joined_elements = np.concatenate((current_elements, previous_elements))
        joined_uncompressed_capacity = current.uncompressed_capacity + previous.uncompressed_capacity
        joined_weights = np.concatenate((current_weights, previous_weights))
        #subsampling seems to be too extreme. Maybe select less aggressively
        #maybe choose combined uncompressed capacity as n which would probably not contradict the chatalic paper
        #breakpoint()
        if self.apply_subsampling and joined_uncompressed_capacity > 1:

            if not self.started_ss :
                self.started_ss = True
                #print(f"started subsampling at calculation of merge to size: {current.uncompressed_capacity * 2}")
            m = math.ceil(math.sqrt(joined_uncompressed_capacity))  # size of the subsample
            #ToDo: uncomment this to sample with replacement
           
            m_idx = np.random.default_rng().integers(len(joined_elements), size=m)
            #m_idx = np.random.default_rng().choice(len(joined_elements), size=m, replace=False)
            #m_idx = range(0,m)
            subsample = joined_elements[m_idx]
            K_z = self.k(subsample, joined_elements)
            #K_m = np.zeros((m, m))  # initialize the kernel matrix with zeros
            #for i in range(m):
                #for j in range(m):
                    # reshape to 2D array as rbf_kernel expects 2D array

            K_m = self.k(subsample, subsample)
            
            K_m_inv = self.inv(K_m)
            #breakpoint()
            new_weights = .5 * K_m_inv @ K_z @ joined_weights
        else:
            m = joined_uncompressed_capacity
            subsample = joined_elements
            new_weights = .5 * joined_weights
       # assuming current_elements and previous_elements have the same length


        
        #breakpoint()
       
        return self.merge_buckets(
            bucket_list[:-2]
            + [
                Bucket(
                    elements=subsample,
                    weights=new_weights,
                    capacity=m,
                    uncompressed_capacity=joined_uncompressed_capacity
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
        if previous.uncompressed_capacity == current.uncompressed_capacity:
            self.buckets = self.buckets[:-2] + [self.merge_buckets(self.buckets[-2:])]
            self._merge()



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
    def __init__(self, elements, capacity, weights, uncompressed_capacity=1):
        """
           A class used to represent a bucket of datapoints.

           Attributes
           ----------
           elements : numpy.ndarray
               A numpy array containing the elements (datapoints) in the bucket. This can be viewed as a vector.
           capacity : int
               The number of elements (datapoints) the bucket holds.
           weights : numpy.ndarray
               A numpy array containing the weights of the elements. These weights indicate the significance or importance
               of each element during the calculation of the Maximum Mean Discrepancy (MMD). This can also be viewed as a vector.
           uncompressed_capacity : int, optional
               The total number of datapoints the bucket represents (default is 1).
               This attribute is particularly useful when the bucket undergoes merging operations which might delete some datapoints,
               but the bucket remains representative of all original datapoints.

        """
        self.elements = elements
        self.capacity = capacity
        self.uncompressed_capacity = uncompressed_capacity
        self.weights = weights

    def __str__(self):
        return f"Elems:\t{self.elements}\nWeights:\t{self.weights}\nUncomp_Cap:\t{self.uncompressed_capacity}"


if __name__ == "__main__":
    bs = BucketStream(gamma=1, compress=False)
    bs.insert(np.array([1]))
    bs.insert(np.array([2]))
    bs.insert(np.array([3]))
    bs.insert(np.array([4]))
    bs.insert(5)
    # bs.insert(6)
    # bs.insert(7)
    # bs.insert(8)
    print(bs)
    print(bs.buckets[0].XX)

## Tests



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
