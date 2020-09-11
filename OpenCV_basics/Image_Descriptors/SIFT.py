import numpy as np
import cv2

img=cv2.imread('images/panorama.PNG')
blur1=cv2.GaussianBlur(img,(3,3),1)
blur2=cv2.GaussianBlur(img,(7,7),1)
cv2.imshow('blur1',blur1)
cv2.imshow('blur2',blur2)
dog=blur2-blur1
cv2.imshow('dog',dog)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift=cv2.xfeatures2d.SIFT_create(8000)
kp,desc = sift.detectAndCompute(gray,None)
img=cv2.drawKeypoints(img,kp,img,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img',img)
cv2.waitKey()