import cv2
import skimage.io
import numpy as np
import BRIEF
import matplotlib.pyplot as plt


image_path = '../data/model_chickenbroth.jpg'
img = skimage.io.imread(image_path)
match_num = np.zeros(37)


for i in range(int(360/10)+1):
    im1 = img
    cols = img.shape[0]
    rows = img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), i*10, 1)
    im2 = cv2.warpAffine(im1, M, (rows, cols))
    locs1, desc1 = BRIEF.briefLite(im1)
    locs2, desc2 = BRIEF.briefLite(im2)
    match_num[i] = BRIEF.briefMatch(desc1, desc2).shape[0]

plt.bar(range(len(match_num)), match_num)
plt.show()
