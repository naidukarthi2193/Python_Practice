import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


img = cv2.imread('kar.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray,(5,5))
thresh = 255/2
maxValue = 255
th, dst = cv2.threshold(blur, thresh, maxValue, cv2.THRESH_BINARY_INV)

edges = cv2.Canny(img, thresh,maxValue)

im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# plt.imshow(draw)
ref_area = 0.0
max_index = 0
for x in range(len(contours)):
    area = cv2.contourArea(contours[x])
    if area > ref_area:
        ref_area = area
        max_index = x
draw1 = cv2.drawContours(img, contours, max_index, (255, 0, 0), 3)
# plt.imshow(draw1)
print(ref_area)
cv2.imshow('image',draw1)
cv2.imshow('dst',dst)
cv2.waitKey(0)
#
# for c in contours:
#     # compute the center of the contour
#     M = cv2.moments(c)
#
#     # calculate x,y coordinate of center
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#     else:
#         cX, cY = 0, 0
#
#
#
#     cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
#     # cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#     # display the image
# cv2.imshow("Image", edges)
#
# cv2.waitKey(0)















    # # show the image
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
#     draw = cv2.drawContours(img, [approx],-1, (0,255,0), 3)


# cnt=contours[0]
# area = cv2.contourArea(cnt)