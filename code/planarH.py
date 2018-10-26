import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''

    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    one = np.ones((1,p2.shape[1]))
    p2 = np.concatenate((p2, one), axis=0)
    A = np.zeros((p2.shape[1]*2,9))
    for i in range(p1.shape[1]):
        A[i * 2, 0:3] = 0*p2[:,i]
        A[i * 2, 3:6] = -1*p2[:,i]
        A[i * 2, 6:9] = p1[1,i]*p2[:,i]
        A[i * 2+1, 0:3] = 1 * p2[:, i]
        A[i * 2+1, 3:6] = 0 * p2[:, i]
        A[i * 2+1, 6:9] = -p1[0, i] * p2[:, i]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    H2to1 = np.reshape(vh[-1,:],(3,3))

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    bestH = np.zeros((3,3))
    best_match_num = 0

    for g in range(num_iter):
        matches = matches.copy()
        initial_index = np.random.randint(0,matches.shape[0],4)
        p1 = np.zeros((2,4))
        p2 = np.zeros((2,4))
        for i, x in enumerate(initial_index):
            p1[0, i] = locs1[matches[x, 0], 0]
            p1[1, i] = locs1[matches[x, 0], 1]
            p2[0, i] = locs2[matches[x, 1], 0]
            p2[1, i] = locs2[matches[x, 1], 1]
        H_interm = computeH(p1,p2)
        correct_index = 0

        p1 = []
        p2 = []

        for i in range(matches.shape[0]):
            point1 = locs1[matches[i, 0]]
            point2 = np.array([locs2[matches[i, 1],0],locs2[matches[i, 1],1],1])
            x = np.dot(H_interm[0],point2)/np.dot(H_interm[2],point2)
            y = np.dot(H_interm[1],point2)/np.dot(H_interm[2],point2)
            dist = (point1[0]-x)**2+(point1[1]-y)**2

            if dist <= tol:
                correct_index = correct_index+1
                p1.append(locs1[matches[i, 0]][0:2])
                p2.append(locs2[matches[i, 1]][0:2])

        p1 = np.transpose(p1)
        p2 = np.transpose(p2)

        if correct_index > best_match_num:
            best_match_num = correct_index
            bestH = computeH(p1,p2)

    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)

    initial_index = np.random.randint(0, matches.shape[0], 4)
    p1 = np.zeros((2, 4))
    p2 = np.zeros((2, 4))
    for i, x in enumerate(initial_index):
        p1[0, i] = locs1[matches[x, 0], 0]
        p1[1, i] = locs1[matches[x, 0], 1]
        p2[0, i] = locs2[matches[x, 1], 0]
        p2[1, i] = locs2[matches[x, 1], 1]
    H_interm = computeH(p1, p2)
    point2 = np.array([p2[0,0],p2[1,0],1])
    x = np.dot(H_interm[0],point2)/np.dot(H_interm[2],point2)
    y = np.dot(H_interm[1],point2)/np.dot(H_interm[2],point2)
    x = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

