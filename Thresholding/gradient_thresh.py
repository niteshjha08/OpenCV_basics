import cv2
import numpy as np

img=cv2.imread('original.PNG')
thresh_range=(0.2,0.9)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
abs_sobelx=np.absolute(sobelx)
norm_x=(255*abs_sobelx/np.max(abs_sobelx)).astype(np.uint8)
abs_sobely=np.absolute(sobely)
norm_y=(abs_sobely/np.max(abs_sobely))
print(norm_x[:10,:10])
print(type(norm_y))
cv2.imshow('sobelx',norm_x)
cv2.imshow('sobely',norm_y)

cv2.waitKey()
