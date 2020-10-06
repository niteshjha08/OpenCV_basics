import cv2
import numpy as np
from math import atan2
import matplotlib.pyplot as plt
img=cv2.imread('original.PNG')
thresh_range=(0.2,0.9)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
abs_sobelx=np.absolute(sobelx)
abs_sobely=np.absolute(sobely)
dir=np.arctan2(abs_sobely,abs_sobelx)
dir_norm=dir/np.max(dir)
print(dir[:10])
dir_thresh=(1.5,3)
binary_op=np.zeros_like(gray)
binary_op[(dir>dir_thresh[0]) & (dir<dir_thresh[1])]=255
cv2.imshow('dir',binary_op)
cv2.waitKey()

