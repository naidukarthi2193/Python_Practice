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

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # read image and resize to 500x500
    img = frame
    img = image_resize(img, width=1080, height=1080)

    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Median Blur to image
    img_median = cv2.medianBlur(img_gray, 5)

    # thresholding
    ret, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # erosion dialtion closing
    erosion = cv2.erode(img_median, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(closing, 9, 220)

    # contour
    img_contours, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # approx ploy line
    # epsilon = 0.1 * cv2.arcLength(contours, True)
    # approx = cv2.approxPolyDP(contours, epsilon, True)

    # drawing all contours
    draw_contours = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # draw ellipse around contour
    if len(contours) != 0:
        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)

    cv2.imshow('frame', draw_contours)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



