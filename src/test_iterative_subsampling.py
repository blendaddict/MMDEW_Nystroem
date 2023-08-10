from mmdew.bucket_stream2 import BucketStream
from mmdew.bucket_stream_old import BucketStream as OldBucketStream
import numpy as np
import pandas as pd
from mmdew.mmd import MMD
mymmd = MMD(biased=True, gamma=1)
from sklearn import metrics
import numpy.linalg as la

X = np.random.normal(0, 1, 32)
Y = np.random.normal(1, 2, 32)


m_idx = np.random.default_rng().integers(32, size=5)
subsample = X[m_idx]
start_weights = ((1/32) * la.pinv(metrics.pairwise.linear_kernel(subsample.reshape(-1,1),subsample.reshape(-1,1)))
                 @ metrics.pairwise.linear_kernel(subsample.reshape(-1,1),X.reshape(-1,1)) @ np.ones(32))

print(f"real mmd:{mymmd.mmd(X.reshape(-1,1),Y.reshape(-1,1))}")

end_weights = np.ones((32,1))/32
addend_1 = start_weights.T @ metrics.pairwise.linear_kernel(subsample.reshape(-1,1), subsample.reshape(-1,1)) @ start_weights
addend_2 = end_weights.T @ metrics.pairwise.linear_kernel(Y.reshape(-1,1), Y.reshape(-1,1)) @ end_weights
addend_3 = start_weights.T @ metrics.pairwise.linear_kernel(subsample.reshape(-1,1), Y.reshape(-1,1)) @ end_weights

print(f" nyström mmd: {(addend_1 + addend_2 - (2 * addend_3))}")

import math

X_1 = np.random.normal(0, 1, 32)
X_2 = np.random.normal(0, 1, 32)
Y = np.random.normal(1, 2, 64)


m_idx1 = np.random.default_rng().integers(32, size=5)
subsample1 = X_1[m_idx1]
m_idx2 = np.random.default_rng().integers(32, size=5)
subsample2 = X_2[m_idx1]
bucket1_weights = ((1/32) * la.pinv(metrics.pairwise.linear_kernel(subsample1.reshape(-1,1),subsample1.reshape(-1,1)))
                 @ metrics.pairwise.linear_kernel(subsample1.reshape(-1,1),X_1.reshape(-1,1)) @ np.ones(32))
bucket2_weights = ((1/32) * la.pinv(metrics.pairwise.linear_kernel(subsample2.reshape(-1,1),subsample2.reshape(-1,1)))
                 @ metrics.pairwise.linear_kernel(subsample2.reshape(-1,1),X_2.reshape(-1,1)) @ np.ones(32))

#getting new elements
joined_subsamples = np.concatenate((subsample1, subsample2))
joined_uncompressed_capacity = 64
m = round(math.sqrt(64))
m_idx_new = np.random.default_rng().integers(len(joined_subsamples), size=m)
new_subsample = joined_subsamples[m_idx_new]

#getting new weights
new_subsample = new_subsample.reshape(-1,1)
joined_subsamples = joined_subsamples.reshape(-1,1)
joined_weights = np.concatenate((bucket1_weights,bucket2_weights))
K_z = metrics.pairwise.linear_kernel(new_subsample, joined_subsamples)
K_m_inv = la.pinv(metrics.pairwise.linear_kernel(new_subsample, new_subsample))
new_weights = .5 * K_m_inv @ K_z @ joined_weights

#weights for Y
end_weights = np.ones((64,1))/64

addend_1 = new_weights.T @ metrics.pairwise.linear_kernel(new_subsample.reshape(-1,1), new_subsample.reshape(-1,1)) @ new_weights
addend_2 = end_weights.T @ metrics.pairwise.linear_kernel(Y.reshape(-1,1), Y.reshape(-1,1)) @ end_weights
addend_3 = new_weights.T @ metrics.pairwise.linear_kernel(new_subsample.reshape(-1,1), Y.reshape(-1,1)) @ end_weights

print(f"real mmd:{mymmd.mmd(np.concatenate((X_1,X_2)).reshape(-1,1),Y.reshape(-1,1))}")
print(f"nyström mmd: {(addend_1 + addend_2 - (2 * addend_3))[0][0]}")
breakpoint()




