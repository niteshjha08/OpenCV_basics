import numpy as np
import cv2

cap=cv2.VideoCapture(0)
#_,img=cap.read()

#cv2.waitKey(3000)
#_,img=cap.read()
#cv2.imwrite('saved.jpg',img)
img=cv2.imread('saved.jpg')
img1=img.copy()
img2=img.copy()
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, imgthreshold=cv2.threshold(imggray,100,255,cv2.THRESH_BINARY_INV)
contours,heirarchy=cv2.findContours(imgthreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
maxcnt=max(contours,key=cv2.contourArea)
print(maxcnt.shape)
print(type(maxcnt))
cv2.drawContours(img1,[maxcnt],0,(0,255,0),2)
epsilon=0.01*cv2.arcLength(maxcnt,True)
cnt=cv2.approxPolyDP(maxcnt,epsilon,True)
cv2.drawContours(img,[cnt],0,(0,0,255),2)
hull=cv2.convexHull(maxcnt)
hullarr=np.array([hull])
cv2.drawContours(img2,[hull],0,(0,0,255),2)

cv2.imshow('contour',img1)
cv2.imshow('approx',img)

cv2.imshow('hull',img2)
print(type(cnt))
cv2.waitKey(0)
#cv2.drawContours(img1, c, -1, (0, 255, 0), 2)
#
# for c in contours:
#     epsilon=0.01*cv2.arcLength(c,True)
#     cnt=cv2.approxPolyDP(c,epsilon,True)
#
# cv2.waitKey(0)