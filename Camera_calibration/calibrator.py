import cv2
import numpy as np

img=cv2.imread('calibration_test.png')
nx=8
ny=6
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners=cv2.findChessboardCorners(gray,(nx,ny))
if (ret==True):
    cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
    print("Entered condition")
cv2.imshow('img',img)
cv2.waitKey()
print("gray size is:",gray.shape[::-1])