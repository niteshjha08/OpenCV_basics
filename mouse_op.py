import numpy as np
import cv2
import time
clicked=False
def myfunc(event,x,y,flags,params):
    global clicked
    if event==cv2.EVENT_LBUTTONDOWN:
        if(x<320 and y<240):print('1 is pressed')
        if (x < 320 and y > 240): print('3 is pressed')
        if (x > 320 and y < 240): print('2 is pressed')
        if(x>320 and y>240):print('4 is pressed')
        clicked=True
cap=cv2.VideoCapture(0)
cv2.namedWindow('window')
_,img=cap.read()
cv2.setMouseCallback('window',myfunc)
time.sleep(5)

while cv2.waitKey(10)==-1 and not clicked:
    _,img=cap.read()
    cv2.imshow('window',img)

