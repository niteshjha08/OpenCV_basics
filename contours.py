import cv2
import numpy as np

cap=cv2.VideoCapture(0)
_,img=cap.read()
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,imgthresh=cv2.threshold(imggray,30,255,cv2.THRESH_BINARY_INV)
contours,_=cv2.findContours(imgthresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
img1=img.copy()
for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

    rect=cv2.minAreaRect(c)
    points=cv2.boxPoints(rect)
    points=np.int0(points)
    img=cv2.drawContours(img,[points],0,(0,0,255),2)

    (x,y),radius=cv2.minEnclosingCircle(c)
    x=np.int0(x)
    y=np.int0(y)
    radius=np.int0(radius)
    cv2.circle(img,(x,y),radius,(0,0,0),2)

imgcontours=cv2.drawContours(img1,contours,-1,(0,255,0),3)
cv2.imshow('contour',imgcontours)
cv2.imshow('imgthresH',imgthresh)
cv2.imshow('img',img)
cv2.imshow('img1',img1)
cv2.imshow('gray',imggray)
cv2.waitKey(0)