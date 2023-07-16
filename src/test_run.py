from mmdew.bucket_stream2 import BucketStream


if __name__ == "__main__":
    X = [[1,0],[2,0],[3,27]]
    bs = BucketStream(gamma=1)
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[0])
    bs.insert(X[1])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])
    bs.insert(X[2])

