import numpy as np
import cv2
import matplotlib.pyplot as plt


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    for i in range(gaussian_pyramid.shape[2]-1):
        DoG_pyramid.append(gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    DoG_levels = levels[1:]

    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    '''
    DoG_pyramid = DoG_pyramid.copy()
    h = DoG_pyramid.shape[0]
    w = DoG_pyramid.shape[1]
    c = DoG_pyramid.shape[2]
    principal_curvature = np.zeros((h,w,c))
    for k in range(c):
        dog_sub = DoG_pyramid[:,:,k]
        ksize = 5
        Dx = cv2.Sobel(dog_sub, -1, 1, 0, ksize)
        Dy = cv2.Sobel(dog_sub, -1, 0, 1, ksize)
        Dxx = cv2.Sobel(Dx, -1, 1, 0, ksize)
        Dxy = cv2.Sobel(Dx, -1, 0, 1, ksize)
        Dyx = cv2.Sobel(Dy, -1, 1, 0, ksize)
        Dyy = cv2.Sobel(Dy, -1, 0, 1, ksize)
        for i in range(h):
            for j in range(w):
                H = np.array([[Dxx[i, j], Dxy[i, j]], [Dyx[i, j], Dyy[i, j]]])
                trace = np.trace(H)
                det = np.linalg.det(H)
                if det == 0:
                    det = 0.00001
                principal_curvature[i, j, k] = (trace ** 2) / det
        principal_curvature = np.clip(principal_curvature, -100, 100 )
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(DoG_pyramid, kernel)
    dilated = cv2.dilate(DoG_pyramid, kernel)
    min = np.where(DoG_pyramid == eroded)
    max = np.where(DoG_pyramid == dilated)
    min = np.array([min[0], min[1], min[2]]).transpose().tolist()
    max = np.array([max[0], max[1], max[2]]).transpose().tolist()


    im_1 = np.transpose(DoG_pyramid, (2, 1, 0))
    kernel_hc = np.array([1, 1, 1]).reshape((3, 1))
    num_layer, det = divmod(im_1.shape[2], 500)
    min_hc_temp = []
    max_hc_temp = []
    index = 0

    # Only have 1~2 loops
    while num_layer>0:
        im_hc = im_1[:, :, index * 500:(index + 1) * 500]
        eroded_hc = cv2.erode(im_hc, kernel_hc)
        dilated_hc = cv2.dilate(im_hc, kernel_hc)
        min_hc = np.where(im_hc == eroded_hc)
        max_hc = np.where(im_hc == dilated_hc)
        min_hc_temp.append(np.array([min_hc[2]+500*index, min_hc[1], min_hc[0]]).transpose())
        max_hc_temp.append(np.array([max_hc[2]+500*index, max_hc[1], max_hc[0]]).transpose())
        index = index + 1
        num_layer = num_layer - 1

    im_hc = im_1[:, :, index * 500:index * 500+det]
    eroded_hc = cv2.erode(im_hc, kernel_hc)
    dilated_hc = cv2.dilate(im_hc, kernel_hc)
    min_hc = np.where(im_hc == eroded_hc)
    max_hc = np.where(im_hc == dilated_hc)

    min_hc_temp.append(np.array([min_hc[2]+500*index, min_hc[1], min_hc[0]]).transpose())
    max_hc_temp.append(np.array([max_hc[2]+500*index, max_hc[1], max_hc[0]]).transpose())


    # Only have 1~2 loops
    min_hc = min_hc_temp[0]
    if len(min_hc_temp)>1:
        for i in range(1,len(min_hc_temp)):
            min_hc = np.concatenate((min_hc,min_hc_temp[i]),axis=0)

    max_hc = max_hc_temp[0]
    if len(max_hc_temp)>1:
        for i in range(1, len(max_hc_temp)):
            max_hc = np.concatenate((max_hc,max_hc_temp[i]),axis=0)

    min_hc = min_hc.tolist()
    max_hc = max_hc.tolist()
    str_min = [str(i) for i in min]
    str_max = [str(i) for i in max]
    str_min_hc = [str(i) for i in min_hc]
    str_max_hc = [str(i) for i in max_hc]


    qq = set(str_min) - (set(str_min)-set(str_min_hc))
    pp = set(str_max) - (set(str_max)-set(str_max_hc))
    q = [eval(i) for i in qq]
    p = [eval(i) for i in pp]
    extrema = p+q

    locsDoG = []
    for i in range(len(extrema)):
        if abs(principal_curvature[extrema[i][0], extrema[i][1], extrema[i][2]]) < th_r and abs(
                DoG_pyramid[extrema[i][0], extrema[i][1], extrema[i][2]]) > th_contrast:
            locsDoG.append([extrema[i][1],extrema[i][0],extrema[i][2]])
    locsDoG = np.array(locsDoG)
    print("find local extrama")

    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    PC = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, PC, th_contrast, th_r)
    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    # test gaussian pyramid

    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)


