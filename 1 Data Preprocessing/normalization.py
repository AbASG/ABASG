import numpy as np

# load data
X = np.load("../datasets/X.npy")    # (3000, 40, 4)
Y = np.load("../datasets/Y.npy")    # (3000, 3)

# 截取有效信息
X = X[:, :, 0:2]                    # (3000, 40, 2)

# normalization method 1
# for i in range(X.shape[2]):
#     _range = np.max(X[:, :, i]) - np.min(X[:, :, i])
#     X[:, :, i] = (X[:, :, i] - np.min(X[:, :, i])) / _range

# normalization method 2
for i in range(X.shape[0]):
    for j in range(X.shape[2]):
        _range = np.max(X[i, :, j]) - np.min(X[i, :, j])
        X[i, :, j] = (X[i, :, j] - np.min(X[i, :, j])) / _range
#print(X)

# 反one-hot
y = np.zeros(Y.shape[0])            # (3000, )
for i in range(Y.shape[0]):
    if Y[i, 0] == 1:
        y[i] = 0
    elif Y[i, 1] == 1:
        y[i] = 1
    else:
        y[i] = 2
y = y.astype(int)

# save data
np.save("../datasets/X_prep.npy", X)
np.save("../datasets/Y_prep.npy", y)