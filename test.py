import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# data_dir = '0916_Data Samples 2/'
data_dir = '0825_DataSamples 1/'
# from keras.preprocessing import image
# x = []
# for file in os.listdir(data_dir):
#     if file.endswith('.json'):
#         continue
#     img = image.load_img(data_dir + file)
#     x.append(img.size)
#
# x = np.array(x)
#
# hist, bin = np.histogram(x[:,1], bins=np.unique(x[:,1]))
# plt.subplot(1, 2, 1)
# plt.hist(x[:,1], bin)
# hist, bin = np.histogram(x[:,0], bins=np.unique(x[:,0]))
# plt.subplot(1, 2, 2)
# plt.hist(x[:,0], bin)
# plt.show()
# print(max(x[:,1]), max(x[:, 0]))

import cv2
from helpers import *
import words
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['image.cmap'] = 'gray'
num_subplot = 3
for file in os.listdir(data_dir)[:]:
    if file.endswith('.json'):
        continue
    img = cv2.imread(data_dir + file, 1)
    plt.subplot(num_subplot, 1, 1)
    plt.imshow(img)
    img = resize(img, 120, always=True)
    # img = cv2.GaussianBlur(img, (11, 11), 71)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.subplot(num_subplot, 1, 2)
    plt.imshow(img)
    # _, img = cv2.threshold(img, 171, 255, cv2.THRESH_BINARY_INV)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 71, 17)
    plt.subplot(num_subplot, 1, 3)
    plt.imshow(img)
    plt.show()
    # boxes = words.detection(img)
