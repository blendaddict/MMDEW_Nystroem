from mmdew.bucket_stream2 import BucketStream
from mmdew.bucket_stream_old import BucketStream as OldBucketStream
import numpy as np
import pandas as pd
from mmdew.mmd import MMD
if __name__ == "__main__":
    bs_ss = BucketStream(gamma=1)
    bs_no_ss = BucketStream(gamma=1, apply_subsampling=False)
    bs_old = OldBucketStream(gamma=1)
    mymmd = MMD(biased=True, gamma=1)

    for exponent in range(1, 11):
        bs_ss = BucketStream(gamma=1)
        bs_no_ss = BucketStream(gamma=1, apply_subsampling=False)
        bs_old = OldBucketStream(gamma=1)
        limit = 2 ** exponent
        X = np.random.normal(0, 1, limit)
        Y = np.random.normal(1, 2, limit - 1)
        #singleX = [1,2,3]
        #singleY = [4,5,6]
        #sample1 = []
        #sample2 = []
        for i in range(0, limit):
            bs_ss.insert_no_cut(X[i])
            #bs_no_ss.insert_no_cut([X[i]])
            #bs_old.insert_no_cut([X[i]])

            #bs_ss.insert_no_cut(singleX)
            #bs_no_ss.insert_no_cut(singleX)
            #sample1 += [singleX]
            #bs_old.insert_no_cut(singleX)
        for i in range(0, limit - 1):
            bs_ss.insert_no_cut(Y[i])
            #bs_no_ss.insert_no_cut([Y[i]])
            #bs_old.insert_no_cut([Y[i]])

            #bs_ss.insert_no_cut(singleY)
            #bs_no_ss.insert_no_cut(singleY)
            #sample2 += [singleY]
            #bs_old.insert_no_cut(singleY)
        #if exponent == 10:
            #breakpoint()

        print(f"SS:    exponent: {exponent} bucket count: {len(bs_ss.buckets)} mmd value={bs_ss.mmd(1)[0]}")
        #print(f"No SS: exponent: {exponent} bucket count: {len(bs_no_ss.buckets)} mmd value={bs_no_ss.mmd(1)[0]}")
        print(f"real MMD: {mymmd.mmd(X.reshape(-1, 1),Y.reshape(-1, 1))}")
        #print(f"real MMD: {mymmd.mmd([singleX,singleX], [singleY])}")
        #print(f"MMDEW: exponent: {exponent} bucket count: {len(bs_old.buckets)} mmd value={bs_old.mmd(1)[0]}\n")
        # print(X)
        # print(Y)