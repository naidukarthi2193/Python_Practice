import cv2
import numpy as np
from matplotlib import pyplot as plt
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
kernel = np.ones((5,5),np.uint8)


#read image and resize to 500x500
img = cv2.imread('img013.jpg')
x=img
img = image_resize(img, width = 500, height = 500)
plt_show(img)
#convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt_show(img_gray)
#Median Blur to image
img_median= cv2.medianBlur(img_gray,5)
plt_show(img_median)
#thresholding
ret, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt_show(otsu)
#erosion dialtion closing
erosion = cv2.erode(img_median,kernel,iterations = 1)
plt_show(erosion)
dilation = cv2.dilate(erosion,kernel,iterations = 5)
plt_show(dilation)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
plt_show(closing)
#Canny edge detection
edges = cv2.Canny(closing,9,220)
plt_show(edges)
#merge minor contours
dilation2 = cv2.dilate(edges,kernel,iterations = 5)
plt_show(dilation2)
#contour
img_contours, contours, hierarchy = cv2.findContours(dilation2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#approx ploy line
# epsilon = 0.1*cv2.arcLength(contours,True)
# approx = cv2.approxPolyDP(contours,epsilon,True)

#drawing all contours
draw_contours= cv2.drawContours(img, contours, -1, (0,255,0), 1)

#draw ellipse around contour
if len(contours) != 0:
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    ellipse = cv2.fitEllipse(c)
    cv2.ellipse(img, ellipse, (0, 0, 255), 2)


print(len(contours))
print(type(hierarchy))
print(hierarchy)

plt_show(x)
plt_show(dilation)
plt_show(dilation2)

plt_show(draw_contours)
plt_show(edges)
