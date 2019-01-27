from sys import exit
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from skimage.morphology import watershed
from skimage.morphology import disk
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import cv2
import numpy as np

# selem=disk(5)


loc='D:\Projects\Python\img055.jpg'


img = cv2.imread(loc)
img_gray = rgb2gray(img)
image = img_as_ubyte(img_gray)
denoised = rank.median(img_gray, disk(2))


gradient = rank.gradient(image, disk(2))

markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]
labels = watershed(gradient,markers)

cv2.imshow("gradient",markers)










cv2.imshow("img",img)
cv2.imshow('gray',img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()






