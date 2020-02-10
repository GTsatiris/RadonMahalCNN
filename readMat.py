import scipy.linalg
import scipy.io
import numpy as np
from skimage.transform import radon
from skimage import io
import math
import os

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data, 0)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal

def calculateMahalMatrix(path):

    # mat = scipy.io.loadmat("Data/a1_s1_t1_skel_K2.mat")
    mat = scipy.io.loadmat(path)
    mydata = mat['S_K2']['world'][0][0]

    # print(type(mydata))
    #
    # for joint in range(mydata.shape[0]):
    #     for coord in range (mydata.shape[1]):
    #         print(mydata[joint][coord][0], end=" ")
    #     print()
    #
    # print()
    #
    # print(mydata[:, :, 0])

    # print(mahalanobis(mydata[1, :, 0], mydata[:, :, 0]))

    mahalMatrix = np.empty([mydata.shape[2], mydata.shape[0]])

    for frame in range(mydata.shape[2]):
        for joint in range(mydata.shape[0]):
            # print(mahalanobis(mydata[12, :, frame], mydata[:, :, frame]))
            mahalMatrix[frame, joint] = mahalanobis(mydata[joint, :, frame], mydata[:, :, frame])

    return mahalMatrix

def calculateRadonDataset(path):
    dirPath = path + '_DATA'
    os.mkdir(dirPath)

    mahalMatrix = calculateMahalMatrix(path)

    for i in range(10, mahalMatrix.shape[0]+1):
        tempMat = mahalMatrix[:i, :]

        # theta = np.linspace(0., 180., max(mahalMatrix.shape), endpoint=False)
        sinogram = radon(tempMat, theta=None, circle=False)

        pad_w = (180 - sinogram.shape[0])/2
        if pad_w == math.floor(pad_w):
            # print("IS FLOAT")
            pad_w = math.floor(pad_w)
            npad = ((pad_w, pad_w), (0, 0))
        else:
            pad_w = math.floor(pad_w)
            npad = ((pad_w, pad_w+1), (0, 0))

        padded_sino = np.pad(sinogram, pad_width=npad, mode='constant', constant_values=0)

        io.imsave(dirPath+'/test_result_{}.png'.format(str(i)), padded_sino)

calculateRadonDataset("Data/a1_s1_t1_skel_K2.mat")