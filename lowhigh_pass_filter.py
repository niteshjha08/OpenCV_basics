import cv2
from scipy import ndimage
import numpy as np

kernel_3x3=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
kernel_5x5=np.array([[-1,-1,-1,-1,-1],[-1,1,2,1,-1],[-1,2,4,2,-1],[-1,1,2,1,-1],[-1,-1,-1,-1,-1]])
img=cv2.imread('sampleimg1.jpg')

#Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#Gy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
#imgx=ndimage.convolve(img,Gx)
#imgy=ndimage.convolve(img,Gy)
#cv2.imshow('imgx',imgx)
#cv2.imshow('imgy',imgy)
#cv2.waitKey(0)
cap=cv2.VideoCapture(0)
imgblur=cv2.GaussianBlur(img,(11,11),0)
blurred=cv2.GaussianBlur(img,(11,11),0)
g_hpf=img-blurred
imgblur=cv2.cvtColor(imgblur,cv2.COLOR_BGR2GRAY)
k3=ndimage.convolve(imgblur,kernel_3x3)
k5=ndimage.convolve(imgblur,kernel_5x5)
cv2.imshow('img',img)
cv2.imshow('k3',k3)
cv2.imshow('k5',k5)
cv2.imshow('g_hpf',g_hpf)
cv2.imshow('blurred',blurred)
cv2.waitKey(0)