import cv2
import numpy as np

img=cv2.imread('images/chelseajersey.PNG',0)
blur1=cv2.GaussianBlur(img,(3,3),1)
blur2=cv2.GaussianBlur(img,(3,3),2)
cv2.imshow('blur1',blur1)
cv2.imshow('blur2',blur2)
dog=blur2-blur1
cv2.imshow('dog',dog)
cv2.waitKey()