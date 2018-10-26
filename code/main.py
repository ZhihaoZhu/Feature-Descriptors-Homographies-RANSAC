import numpy as np
import keypointDetect
import skimage.io
import cv2
from PIL import Image, ImageDraw
import BRIEF
import os
import matplotlib.pyplot as plt
import planarH
from scipy.signal import argrelextrema
import datetime


#
# #keypointDetect.py

# ../data/model_chickenbroth.jpg
image_path = '../data/pf_desk.jpg'
#
image_path = '../data/model_chickenbroth.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# newX = img.shape[1]/2
# newY = img.shape[0]/2
# # newX = 500
# # newY = 500
# img = cv2.resize(img, (int(newX),int(newY)))
print(img.shape)
#Get gaussian pyramid
im_pyramid = keypointDetect.createGaussianPyramid(img, sigma0=1,
        k=np.sqrt(2), levels=[-1,0,1,2,3,4])

#Get DoG pyramid

DoG_pyramid, DoG_levels = keypointDetect.createDoGPyramid(im_pyramid, [-1,0,1,2,3,4])

#Get  pricipal curveture

PC = keypointDetect.computePrincipalCurvature(DoG_pyramid)

locsDoG = keypointDetect.getLocalExtrema(DoG_pyramid, DoG_levels, PC, 0.03, 12)
print(locsDoG.shape)
# locsDoG_cr = keypointDetect.getLocalExtrema_cr(DoG_pyramid, DoG_levels, PC, 0.03, 12)
#
# np.save("../test/locs.npy", locsDoG)
# np.save("../test/locs_cr.npy", locsDoG_cr)
#
# print(locsDoG.shape)
# print(locsDoG_cr.shape)


'''
    Visualize
'''

fig = plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.plot(locsDoG[:,0], locsDoG[:,1], 'g.')
plt.draw()
plt.waitforbuttonpress(0)
plt.close(fig)






# if len(img.shape) == 3:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img = np.stack((img,img),axis=-1)
#
# # avatar = Image.fromarray(img)
# avatar = Image.open(image_path)
#
# drawAvatar = ImageDraw.Draw(avatar)
# # drawAvatar.point((10, 100),fill=(255,0,0))
# for i in range(locsDoG.shape[0]):
#     drawAvatar.point((locsDoG[i,1], locsDoG[i,0]),fill=(255,0,0))
#
# del drawAvatar
# avatar.show()

# im = cv2.imread('../data/model_chickenbroth.jpg')

# Get the local extrema in (x,y) plane
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# eroded = cv2.erode(im, kernel)
# dilated = cv2.dilate(im, kernel)
# min = np.where(im == eroded)
# max = np.where(im == dilated)
# min = np.array([min[0], min[1], min[2]]).transpose().tolist()
# max = np.array([max[0], max[1], max[2]]).transpose().tolist()
#
#
#
# im_hc = np.transpose(im, (2, 1, 0))
# kernel_hc = np.array([1,1,1]).reshape((3,1))
# eroded_hc = cv2.erode(im_hc, kernel_hc)
# dilated_hc = cv2.dilate(im_hc, kernel_hc)
# min_hc = np.where(im_hc == eroded_hc)
# max_hc = np.where(im_hc == dilated_hc)
# min_hc = np.array([min_hc[2], min_hc[1], min_hc[1]]).transpose().tolist()
# max_hc = np.array([max_hc[2], max_hc[1], max_hc[1]]).transpose().tolist()
#
# extrema = []
# for p in min:
#     if p in min_hc:
#         extrema.append(p)
# for q in max:
#     if q in max_hc:
#         extrema.append(q)
# extrema = np.array(extrema)
#
# for i in len(extrema):
#     if abs(principal_curvature[extrema[i][0],extrema[i][1],extrema[i][2]])>th_r or abs(DoG_pyramid[extrema[i][0],extrema[i][1],extrema[i][2]])>th_contrast:
#         del extrema[i]




# with open('../data/sphere.txt', "r") as f:
#     str = f.read()
# lines = str.split('\n')
# x = lines[0].split('  ')
# y = lines[1].split('  ')
# z = lines[2].split('  ')
# new_list = []
# for i in range(1, len(x)):
#     new_list.append([float(x[i]), float(y[i]), float(z[i])])
#
# W2 = np.array(new_list).transpose()
# W2_one = np.ones((1, W2[1]))
# W2 = np.concatenate((W2, W2_one), axis=0)

# print(divmod(6,2))
#
# im_hc = np.ones((5,5,512))
# kernel_hc = np.array([1, 1, 1]).reshape((3, 1))
# eroded_hc = cv2.erode(im_hc, kernel_hc)

# x = np.arange(1000000)
# my_list = [str(i) for i in x]
# print(my_list)
#
# x = np.load("../results/q61.npz")
# y = np.load("../results/q62.npz")
#
# locs1_cr = x["locs1"]
# locs1_me = y["locs1"]
# print(np.array_equal(locs1_cr,locs1_me))

# '''
#     Test whether two are identical
# '''
# x = np.load("../test/locs.npy")
# y = np.load("../test/locs_cr.npy")
#
# print(x.shape)
# print(y.shape)
#
# # x = x.transpose()
# # y = y.transpose()
# x = x.tolist()
# y = y.tolist()
#
# x = [str(i) for i in x]
# y = [str(i) for i in y]
#
# # print(x)
# # print(y)
#
# x_set = set(x)
# y_set = set(y)
#
# p = y_set - x_set
# print(p)
# print(x_set==y_set)

