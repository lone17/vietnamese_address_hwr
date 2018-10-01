# -*- coding: utf-8 -*-
"""
Detect words on the page
return array of words' bounding boxes
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from helpers import *

def detection(image, join=True):
    """ Detecting the words bounding boxes """
    image = resize(image, 400, always=True)
    # Preprocess image for word detection
    blurred = cv2.GaussianBlur(image, (13, 13), 17)
    # blurred = image
    plt.subplot(4, 1, 1)
    implt(blurred, cmp=None, t='blurred')
    edgeImg = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # edgeImg = edgeDetect(blurred)
    plt.subplot(4, 1, 2)
    implt(edgeImg, cmp=None, t='edgeImg')
    # ret, edgeImg = cv2.threshold(edgeImg, 150, 255, cv2.THRESH_BINARY_INV)
    edgeImg = cv2.adaptiveThreshold(edgeImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 251, 13)
    plt.subplot(4, 1, 3)
    implt(edgeImg, cmp=None, t='edgeImg')
    bwImage = cv2.morphologyEx(edgeImg, cv2.MORPH_CLOSE,
                               np.ones((11,11), np.uint8))
    # bwImage = edgeImg
    plt.subplot(4, 1, 4)
    implt(bwImage, t='bwImage')
    plt.show()
    # Return detected bounding boxes
    return textDetect(bwImage, edgeImg, join)


def edgeDetect(im):
    """
    Edge detection
    Sobel operator is applied for each image layer (RGB)
    """
    return np.max(np.array([sobelDetect(im[:,:, 0]),
                            sobelDetect(im[:,:, 1]),
                            sobelDetect(im[:,:, 2])]), axis=0)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(im, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv2.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(im, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def sobelDetect(channel):
    """ Sobel operator """
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)

def area(a):
    return a[2] * a[3]

def isPunctuation(a):
    return a[2] < 70 or a[3] < 70

def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return [x, y, w, h]

def isIntersect(a, b, expanded):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = x - min(a[0]+a[2], b[0]+b[2])
    h = y - min(a[1]+a[3], b[1]+b[3])

    if w < 100 and isPunctuation(b):
        return True

    if w > 70 or h > 70:
        return False

    # if not isPunctuation(b) and expanded and w > 20:
    #     return False

    # if (w < -30 and )

    return True

def groupRectangles(rec):
    """
    Uion intersecting rectangles
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles
    """
    tested = [False]*len(rec)
    expanded = [False]*len(rec)
    final = []
    i = 0
    rec = sorted(rec, key=lambda x: x[0])

    while i < len(rec):
        if not tested[i]:
            j = i+1
            while j < len(rec):
                if not tested[j] and isIntersect(rec[i], rec[j], expanded[i]):
                    rec[i] = union(rec[i], rec[j])
                    if isPunctuation(rec[j]):
                        expanded[i] = True
                    tested[j] = True
                    # j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final


def textDetect(bwImage, image, join):
    """ Text detection using contours """

    # Finding contours
    mask = np.zeros(bwImage.shape, np.uint8)
    im2, cnt, hierarchy = cv2.findContours(np.copy(bwImage),
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    boundingBoxes = np.array([0,0,0,0])
    bBoxes = []

    # image for drawing bounding boxes
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Go through all contours in top level
    while (index >= 0):
        x,y,w,h = cv2.boundingRect(cnt[index])
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y+h, x:x+w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)

        # Limits for text
        if r > 0.1 and w > 20 and h > 30:
            bBoxes += [[x, y, w, h]]

        index = hierarchy[0][index][0]

    # Need more work
    if join:
        bBoxes = groupRectangles(bBoxes)
    for (x, y, w, h) in bBoxes:
        cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 2)
        boundingBoxes = np.vstack((boundingBoxes,
                                   np.array([x, y, x+w, y+h])))

    plt.imshow(image)
    plt.show()

    # bBoxes = boundingBoxes.dot(ratio(image, bwImage.shape[0])).astype(np.int64)
    return bBoxes[1:]


def textDetectWatershed(img):
    """ Text detection using watershed algorithm - NOT IN USE """
    # According to: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    img = resize(img, 2000)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,
                                 0.01*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers += 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    implt(markers, t='Markers')
    image = img.copy()

    for mark in np.unique(markers):
        # mark == 0 --> background
        if mark == 0:
            continue

        # Draw it on mask and detect biggest contour
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == mark] = 255

        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # Draw a bounding rectangle if it contains text
        x,y,w,h = cv2.boundingRect(c)
        cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y+h, x:x+w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)

        # Limits for text
        if r > 0.2 and 2000 > w > 15 and 1500 > h > 15:
            cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 2)

    implt(image)
