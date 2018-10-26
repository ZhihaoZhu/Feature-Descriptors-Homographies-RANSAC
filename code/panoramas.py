import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''

    length_expand = 200
    out_size = (im1.shape[1]+length_expand,im1.shape[0])
    im_right = cv2.warpPerspective(im2, H2to1, out_size)
    im_left = cv2.copyMakeBorder(im1, 0, 0, 0, 50, cv2.BORDER_CONSTANT, value=[0,0,0])

    # cv2.imshow('../results/6_1.jpg', im_right)
    # cv2.imwrite('../results/6_1.jpg', im_right)
    # cv2.waitKey(0)

    for i in range(im_left.shape[0]):
        for j in range(im_left.shape[1]):
            if np.array_equal(im_left[i,j],[0,0,0]) and np.array_equal(im_right[i,j],[0,0,0]):
                im_right[i,j] = [0,0,0]
            else:
                if np.array_equal(im_right[i, j], [0, 0, 0]):
                    im_right[i, j] = im_left[i, j]
                else:
                    bl, gl, rl = im_left[i, j]
                    bw, gw, rw = im_right[i, j]
                    im_right[i, j] = [max(bl, bw), max(gl, gw), max(rl, rw)]

    pano_im = im_right

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    '''

    H = im2.shape[0]
    W = im2.shape[1]


    P1 = np.array([[0,0,1],[W,0,1],[0,H,1],[W,H,1]])
    P2 = np.zeros((4,2))
    for i in range(4):
        P2[i, 0] = np.dot(H2to1[0], P1[i]) / np.dot(H2to1[2], P1[i])
        P2[i, 1] = np.dot(H2to1[1], P1[i]) / np.dot(H2to1[2], P1[i])

    W_pano_1 = int(np.max(P2[:,0]))
    H_pano_1 = int(np.max(P2[:,1])-np.min(P2[:,1]))

    W_pano = W
    H_pano = int(W*H_pano_1/W_pano_1)

    s1 = W_pano/W_pano_1
    s2 = H_pano/H_pano_1
    t1 = 0
    t2 = -np.min(P2[:,1])*s2
    M = np.array([[s1, 0, t1],[0, s2, t2],[0,0,1]])

    out_size = (W_pano, H_pano)
    im_left = cv2.warpPerspective(im1, M, out_size)

    im_right = cv2.warpPerspective(im2, np.matmul(M, H2to1), out_size)
    for i in range(im_left.shape[0]):
        for j in range(im_left.shape[1]):
            if np.array_equal(im_left[i,j],[0,0,0]) and np.array_equal(im_right[i,j],[0,0,0]):
                im_right[i,j] = [0,0,0]
            else:
                bl, gl, rl = im_left[i, j]
                bw, gw, rw = im_right[i, j]
                im_right[i, j] = [max(bl,bw),max(gl,gw),max(rl,rw)]

    pano_im = im_right

    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, 5000, 2)

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png', cv2.IMREAD_COLOR)
    im2 = cv2.imread('../data/incline_R.png', cv2.IMREAD_COLOR)

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)

    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save("../results/q6_1.npy", H2to1)
    #
    # pano_im = imageStitching(im1, im2, H2to1)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    # cv2.imwrite('../results/q6_3.jpg', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


