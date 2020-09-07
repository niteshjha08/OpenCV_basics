import cv2
from scipy import ndimage
import numpy as np

cap=cv2.VideoCapture(0)
img=cv2.imread('sampleimg1.jpg')
imgblur=cv2.medianBlur(img,11)
imggray=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)
edgeimg=cv2.Laplacian(imggray,cv2.CV_8U,ksize=5)
cannyimg=cv2.Canny(img,50,70)
invimg=255-edgeimg
cv2.imshow('edgeimg',edgeimg)
cv2.imshow('canny',cannyimg)
cv2.imshow('invimg',invimg)
cv2.waitKey(0)
