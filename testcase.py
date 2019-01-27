import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

img = cv2.imread('img012.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (3, 3))
thresh = 255 / 2
maxValue = 255
th, dst = cv2.threshold(blur, thresh, maxValue, cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
img_cx, img_cy = img[:, :, 0].shape
if img_cx > img_cy:
    ref_dist = img_cx
else:
    ref_dist = img_cy
img_cx = img_cx / 2
img_cy = img_cy / 2
ref_x = 0

for x in range(len(contours)):
    # compute the center of the contour
    M = cv2.moments(contours[x])
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    dist = math.hypot(img_cx - cX, img_cy - cY)
    if dist < ref_dist:
        ref_dist = dist
        ref_x = x
draw1 = cv2.drawContours(img, contours[ref_x], -1, (255, 0, 0), 3)
cv2.imshow("Image", draw1)
cv2.waitKey(0)









