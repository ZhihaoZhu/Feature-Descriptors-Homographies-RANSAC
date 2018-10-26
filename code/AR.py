import cv2
import numpy as np
from planarH import computeH
import matplotlib.pyplot as plt
import skimage.io

def compute_extrinsics(K, H):
    H1 = np.dot(np.linalg.inv(K),H)
    x = H1[:,0:2]
    U, L, Vh = np.linalg.svd(x, full_matrices=True)
    y = np.array([[1,0],[0,1],[0,0]])
    w_12 = np.dot(U,y)
    w_12 = np.dot(w_12,Vh)
    w_3 = np.cross(w_12[:,0],w_12[:,1]).reshape((3,1))
    w = np.concatenate((w_12,w_3),axis=1)
    if np.linalg.det(w) == -1:
        w[:,2] = w[:,2]*(-1)
    sum = 0
    for i in range(3):
        for j in range(2):
            sum = sum+H1[i,j]/w[i,j]
    lamda = sum/6
    print(lamda)
    R = w
    t = H1[:, 2]/lamda
    t = t.reshape((3,1))
    return R,t

def project_extrinsics(K, W, R, t):
    extrinsic = np.concatenate((R,t), axis=1)
    intrinsic_extrinsic = np.dot(K,extrinsic)
    x = np.dot(intrinsic_extrinsic, W)
    X = np.zeros((2, W.shape[1]))
    for i in range(x.shape[1]):
        X[0, i] = x[0, i] / x[2, i]
        X[1, i] = x[1, i] / x[2, i]
    X = X.astype(int)
    return X


if __name__ == '__main__':
    W = np.array([[0,18.2,18.2,0],[0,0,26,26],[0,0,0,0]])
    X = np.array([[483, 1704, 2175, 67], [810,781,2217,2286]])
    K = np.array([[3043.72,0,1196],[0,3043.72,1604],[0,0,1]])
    W1 = W[0:2,:]
    H = computeH(X,W1)
    R,t = compute_extrinsics(K, H)
    with open('../data/sphere.txt', "r") as f:
        str = f.read()
    lines = str.split('\n')
    x = lines[0].split('  ')
    y = lines[1].split('  ')
    z = lines[2].split('  ')
    new_list = []
    for i in range(1, len(x)):
        new_list.append([float(x[i]), float(y[i]), float(z[i])])
    W2 = np.array(new_list).transpose()
    W2_one = np.ones((1, W2.shape[1]))
    W2 = np.concatenate((W2, W2_one), axis=0)
    x = project_extrinsics(K, W2, R, t)+np.array((350,820)).reshape((2,1))
    im = skimage.io.imread('../data/prince_book.jpg')
    fig = plt.figure()
    plt.imshow(im)
    plt.plot(x[0, :], x[1, :], 'y.', markersize=1)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


