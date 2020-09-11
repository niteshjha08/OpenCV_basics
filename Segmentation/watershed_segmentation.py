import cv2
import numpy as np

img=cv2.imread('planets.PNG')
img1=img.copy()
img1=img[20:200,30:530,:]
imggray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#ret,thresh=cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
ret,thresh=cv2.threshold(imggray,150,255,cv2.THRESH_BINARY)
kernel=np.ones((5,5))
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
sure_bg=cv2.dilate(opening,kernel,iterations=3)
dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,3)
dist_matrix=dist_transform.copy().round().astype('uint8')
sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)[1]
sure_fg=sure_fg.astype('uint8')
unknown=cv2.subtract(sure_bg,sure_fg)
ret,markers=cv2.connectedComponents(sure_fg)
markers=markers+1
markers[unknown==255]=0
print(img.shape)
markers=cv2.watershed(img1,markers)
img1[markers==-1]=[0,0,255]
cv2.imshow('unknwon',unknown)
cv2.imshow('opening',opening)
cv2.imshow('surebg',sure_bg)
cv2.imshow('surefg',sure_fg)
cv2.imshow('dist',dist_matrix)
cv2.imshow('thresh',thresh)
cv2.imshow('img',img)
cv2.imshow('copy',img1)
cv2.imshow('markers',markers.astype('uint8'))
cv2.waitKey(0)