import scipy as sp
import scipy.io
import numpy as np

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

mat = scipy.io.loadmat("Data/a1_s1_t1_skel_K2.mat")

mydata = mat['S_K2']['world'][0][0]

print(type(mydata))

for joint in range(mydata.shape[0]):
    for coord in range (mydata.shape[1]):
        print(mydata[joint][coord][0], end=" ")
    print()

print()

print(mydata[:, :, 0])