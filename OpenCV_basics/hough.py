import cv2
import numpy as np

cap=cv2.VideoCapture(0)
_,img=cap.read()
planets=cv2.imread('planets.PNG')

grayplanet=cv2.cvtColor(planets,cv2.COLOR_BGR2GRAY)
_,img=cap.read()
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(imggray,50,70)
minlength=20
maxgap=5
line=cv2.HoughLinesP(edge,1,np.pi/180,120,0,minLineLength=minlength,maxLineGap=maxgap)
for i,j,k,l in line[0]:
    cv2.line(img,(i,j),(k,l),(0,255,0),2)
circles=cv2.HoughCircles(grayplanet,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=0,maxRadius=0)
print(circles)
for i in circles[0]:
    cv2.circle(planets,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)
cv2.imshow('edges',edge)
cv2.imshow('img',img)
cv2.imshow('planets',planets)
cv2.imshow('grayplanet',grayplanet)
cv2.waitKey(0)