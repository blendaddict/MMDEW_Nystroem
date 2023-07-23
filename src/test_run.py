from mmdew.bucket_stream2 import BucketStream


if __name__ == "__main__":
    X = [[1,0],[2,0],[3,27]]
    bs = BucketStream(gamma=1)
    for i in range(0,50):
        bs.insert(X[0])
        bs.insert(X[1])
    for i in range(0,100):
        bs.insert(X[2])
    print(bs.get_changepoints())