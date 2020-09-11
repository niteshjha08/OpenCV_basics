import cv2
import numpy as np
from math import ceil
cap=cv2.VideoCapture(0)

def onMouse(event,x,y,flags,param):
    global started,ended,leftcorner_x,leftcorner_y,rightcorner_x,rightcorner_y
    if(event==cv2.EVENT_LBUTTONDOWN):
        leftcorner_x=x
        leftcorner_y = y
        started=True
    if(event==cv2.EVENT_LBUTTONUP):
        rightcorner_x=x
        rightcorner_y=y
        ended=True
    #if(started==True):
        #cv2.rectangle(img,(leftcorner_x,leftcorner_y),(x,y),(0,0,255),2)

if __name__=="__main__":
    started = False
    ended = False
    (leftcorner_x,leftcorner_y)=(0,0)
    (rightcorner_x,rightcorner_y)=(0,0)
    img=cv2.imread('saved.jpg')
    img1=img.copy()
    cv2.namedWindow('window')
    cv2.setMouseCallback('window',onMouse)
    cv2.imshow('window',img)
    while(cv2.waitKey(10)!=ord('q')):
        if(ended):
            cv2.rectangle(img1, (leftcorner_x, leftcorner_y), (rightcorner_x,rightcorner_y), (0, 0, 255), 2)
            fgmodel=np.zeros([1,65]).astype('float64')
            bgmodel=np.zeros([1,65]).astype('float64')
            mask=np.zeros(img.shape[:2]).astype('uint8')
            print(leftcorner_x,leftcorner_y,rightcorner_x-leftcorner_x,rightcorner_y-leftcorner_y)
            rect=(leftcorner_x,leftcorner_y,rightcorner_x,rightcorner_y)
            cv2.grabCut(img,mask,rect,bgmodel,fgmodel,10,cv2.GC_INIT_WITH_RECT)
            #cv2.imshow('mask',mask1)
            mask2=np.where((mask==0)|(mask==2),0,1).astype('uint8')
            img=img*mask2[:,:,np.newaxis]
            cv2.imshow('grabcut',img)
            cv2.imshow('original',img1)
