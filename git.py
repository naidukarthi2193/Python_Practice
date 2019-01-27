import cv2
import numpy as np
import time
import smtplib
from matplotlib import pyplot as plt

cv2.startWindowThread()
def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
#     plt.imshow(image, cmap="Greys_r")
#     plt.imshow(image, cmap=plt.cm.Spectral)
    plt.imshow(image, cmap=plt.cm.Greys_r)
    plt.show()

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

r_image1 = cv2.imread('img032.jpg')
r_image2 = image_resize(r_image1, width = 500, height = 500)
plt_show(r_image2)
plt.title("Pothole Image")
plt.imshow(r_image2)
plt.show()
#resize_image = cv2.resize(r_image1, (275,180))
#plt_show(resize_image)
#im = cv2.imread('index4.jpg')
im = r_image2
plt_show(im)
# Convert the GrayScale
gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt_show(gray1)
# save the image
cv2.imwrite('grayImg.jpg', gray1)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
plt_show(imgray)
ret,thresh = cv2.threshold(imgray,127,255,0)
image1, contours1, hierarchy1 = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
plt_show(image1)
image2, contours2, hierarchy2 = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
plt_show(image2)
img2 = im.copy()
plt_show(img2)
out = cv2.drawContours(img2, contours2, -1, (0,250,0),1)
plt.title("drawContours Pothole Image")
plt.imshow(out)
plt.show()

plt.imshow(img2)
plt.show()
# cv2.imshow('img1',img2)
# cv2.waitKey(0)
plt.subplot(331),plt.imshow(im),plt.title('GRAY')
plt.xticks([]), plt.yticks([])
img = cv2.imread('img055.jpg',0)
# plt_show(img)
# plt.imshow(img)
# plt.show()
ret,thresh = cv2.threshold(img,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh, 1, 2)
print(type(contours))


cnt = contours[0]

M = cv2.moments(cnt)
print(M)
perimeter = cv2.arcLength(cnt,True)
print (perimeter)
area = cv2.contourArea(cnt)
print (area)
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
print (epsilon)
print (approx)
for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100: continue
    # print cv2.contourArea(c)
    x, y, w, h = rect
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 8)
    cv2.putText(img2, 'Moth Detected', (x + w + 40, y + h), 0, 2.0, (0, 255, 0))

    plt.title("Moth Detected Pothole Image")
    plt.imshow(img2)
    plt.show()
cv2.imshow("Show", img2)
# cv2.imshow('img' , resize_img)
x = cv2.waitKey(0)
if x == 27:
    cv2.destroyWindow('img')
cv2.waitKey()
cv2.destroyAllWindows()
k = cv2.isContourConvex(cnt)
print(k)
blur = cv2.blur(im,(5,5))
plt_show(blur)
gblur = cv2.GaussianBlur(im,(5,5),0)
plt_show(gblur)
median = cv2.medianBlur(im,5)
plt_show(median)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(median,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 5)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(dilation,9,220)
plt.subplot(332),plt.imshow(blur),plt.title('BLURRED')
plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(gblur),plt.title('guassianblur')
plt.xticks([]), plt.yticks([])

plt.subplot(334),plt.imshow(median),plt.title('Medianblur')
plt.xticks([]), plt.yticks([])

plt.subplot(337),plt.imshow(img,cmap = 'gray')
plt.title('dilated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(erosion),plt.title('EROSION')
plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(closing),plt.title('closing')
plt.xticks([]), plt.yticks([])
plt.show()

