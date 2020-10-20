import scipy.linalg
import scipy.io
import numpy as np
from skimage.transform import radon
from skimage import io
import math
import os
import time

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data, 0)
    if not cov:
        cov = np.cov(data.T)
    try:
        inv_covmat = scipy.linalg.inv(cov)
    except np.linalg.LinAlgError:
        noise = np.random.normal(0, .001, cov.shape)
        covDirty = cov + noise
        inv_covmat = scipy.linalg.inv(covDirty)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal

def calculateMahalMatrix(path):

    # mat = scipy.io.loadmat("Data/a1_s1_t1_skel_K2.mat")
    data = np.load(path, allow_pickle=True).item()
    mydata = data['skel_body0']

    # print(mydata.shape)
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

    mahalMatrix = np.empty([mydata.shape[0], mydata.shape[1]])

    for frame in range(mydata.shape[0]):
        for joint in range(mydata.shape[1]):
            # print(mahalanobis(mydata[12, :, frame], mydata[:, :, frame]))
            mahalMatrix[frame, joint] = mahalanobis(mydata[frame, joint, :], mydata[frame, :, :])

    return mahalMatrix

def calculateRadonDataset(path):

    # if '_NPZ' in path or 'PNG_SMPLE' in path:
    #     return

    # print(path)
    # clNum = int(path.split("A", 1)[1][:3])
    #
    # coolNumbers = [23, 7, 94, 12, 50, 10]
    #
    # if clNum not in coolNumbers:
    #     return

    # dirPath = path + '_NPZ'
    # os.mkdir(dirPath)

    mahalMatrix = calculateMahalMatrix(path)
    print(mahalMatrix.shape)

    for i in range(5, mahalMatrix.shape[0]+1):
        tempMat = mahalMatrix[:i, :]

        # theta = np.linspace(0., 180., max(mahalMatrix.shape), endpoint=False)
        sinogram = radon(tempMat, theta=None, circle=False)

        pad_w = (180 - sinogram.shape[0])/2

        if(pad_w >= 0):
            if pad_w == math.floor(pad_w):
                pad_w = math.floor(pad_w)
                npad = ((pad_w, pad_w), (0, 0))
            else:
                pad_w = math.floor(pad_w)
                npad = ((pad_w, pad_w+1), (0, 0))

            padded_sino = np.pad(sinogram, pad_width=npad, mode='constant', constant_values=0)

            # np.savez(dirPath+'/frame_{}'.format(str(i)), sino = padded_sino, clNum = clNum)
            # io.imsave(dirPath+'/frame_{}.png'.format(str(i)), padded_sino)

        else:
            if pad_w == math.ceil(pad_w):
                pad_w = math.floor(pad_w)
                npad_1 = abs(pad_w)
                npad_2 = abs(pad_w)
            else:
                pad_w = math.floor(pad_w)
                npad_1 = abs(pad_w)
                npad_2 = abs(pad_w) - 1

            cropped_sino = sinogram[npad_1:sinogram.shape[0]-npad_2, :]

            # np.savez(dirPath+'/frame_{}'.format(str(i)), sino = cropped_sino, clNum = clNum)
            # io.imsave(dirPath + '/frame_{}.png'.format(str(i)), cropped_sino)

# start = time.time()
# calculateMahalMatrix("NTURGB-D_120_Code/Python/raw_npy/S032C003P106R002A094.skeleton.npy")
# end = time.time()
# print(end - start)
# # calculateRadonDataset("Data/a1_s2_t1_skel_K2.mat")
directory = os.listdir('NTURGB-D_120_Code/Python/raw_npy')
allNum = 0
for fname in directory:
    if '_NPZ' in fname or 'PNG_SMPLE' in fname:
        continue

    # print(path)
    clNum = int(fname.split("A", 1)[1][:3])

    coolNumbers = [23, 7, 94, 12, 50, 10]

    if clNum not in coolNumbers:
        continue

    allNum = allNum + 1

print('All files: ', allNum)
print('Creating Radon Dataset')
num = 1
# f1 = open('average_times_jit.txt', "a")
for fname in directory:
    if '_NPZ' in fname or 'PNG_SMPLE' in fname:
        continue

    # print(path)
    clNum = int(fname.split("A", 1)[1][:3])

    coolNumbers = [23, 7, 94, 12, 50, 10]

    if clNum not in coolNumbers:
        continue

    start = time.time()
    calculateRadonDataset("NTURGB-D_120_Code/Python/raw_npy/"+fname)
    end = time.time()
    typeStr = 'Finished ' + str(num) + '/' + str(allNum) + ' in ' + str(end - start) + 'seconds'
    # print(typeStr, end="\r")
    print(typeStr)
    num = num + 1

    # f1.write('********************************************************\n')
    # f1.write('Average time for mahal matrix per frame: {}\n'.format(str(mahalFrameTime / framecount1)))
    # f1.write('Average time for srf calculation per frame: {}\n'.format(str(srfFrameTime / framecount2)))
    # # f1.write('Frame count 1: {}\n'.format(str(framecount1)))
    # f1.write('Frame count 2: {}\n'.format(str(framecount2)))
    # f1.flush()
# f1.write('********************************************************\n')
# f1.write('Average time for mahal matrix per frame: {}\n'.format(str(mahalFrameTime / framecount1)))
# f1.write('Average time for srf calculation per frame: {}\n'.format(str(srfFrameTime / framecount2)))
# f1.write('Frame count 1: {}\n'.format(str(framecount1)))
# f1.write('Frame count 2: {}\n'.format(str(framecount2)))
# f1.close()
#
# mat = scipy.io.loadmat("Data/a1_s1_t1_skel_K2.mat")
# # mat = scipy.io.loadmat(path)
# mydata = mat['S_K2']['world'][0][0]
#
# # print(type(mydata))
# #
# # for joint in range(mydata.shape[0]):
# #     for coord in range (mydata.shape[1]):
# #         print(mydata[joint][coord][0], end=" ")
# #     print()
# #
# # print()
# #
# # print(mydata[:, :, 0])
#
# # print(mahalanobis(mydata[1, :, 0], mydata[:, :, 0]))
#
# mahalMatrix = np.empty([mydata.shape[2], mydata.shape[0]])
#
# for frame in range(mydata.shape[2]):
#     for joint in range(mydata.shape[0]):
#         # print(mahalanobis(mydata[12, :, frame], mydata[:, :, frame]))
#         mahalMatrix[frame, joint] = mahalanobis(mydata[joint, :, frame], mydata[:, :, frame])
#
# print(mahalMatrix[0])

# calculateMahalMatrix("Data/a1_s2_t1_skel_K2.mat")