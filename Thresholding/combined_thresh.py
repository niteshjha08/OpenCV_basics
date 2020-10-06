import cv2
import numpy as np

def abs_sobel_thresh(img,orient='x',thresh=(0,255),ksize=3):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if(orient=='x'):
        x=1
        y=0
    else:
        x=0
        y=1
    sobel=cv2.Sobel(gray,cv2.CV_64F, x,y)
    abs_sobel=np.absolute(sobel)
    norm_sobel=(255*abs_sobel/np.max(abs_sobel)).astype(np.uint8)
    binary_op=np.zeros_like(gray)
    binary_op[(norm_sobel>thresh[0]) & (norm_sobel<thresh[1])]=255
    return binary_op

def mag_sobel(img,thresh=(0,255),ksize=3):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx=cv2.Sobel(gray,cv2.CV_64F, 1,0)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1)

    abs_sobel_xy=np.sqrt(sobelx**2+sobely**2)
    norm_sobel=(255*abs_sobel_xy/np.max(abs_sobel_xy)).astype(np.uint8)
    binary_op=np.zeros_like(gray)
    binary_op[(norm_sobel>thresh[0]) & (norm_sobel<thresh[1])]=255
    return binary_op

def dir_sobel(img,thresh=(0,255),ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel= np.arctan2(abs_sobely,abs_sobelx)
    norm_sobel = (255 * dir_sobel / np.max(dir_sobel)).astype(np.uint8)
    binary_op = np.zeros_like(gray)
    binary_op[(norm_sobel > thresh[0]) & (norm_sobel < thresh[1])] = 255
    return binary_op
img=cv2.imread('signs_vehicles_xygrad.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gradx=abs_sobel_thresh(img,orient='x',thresh=(30,100))
grady=abs_sobel_thresh(img,orient='y',thresh=(30,100))
mag=mag_sobel(img,(30,100))
dir=dir_sobel(img,(30,100))
final=np.zeros((img.shape[0],img.shape[1]))
print(final.shape)
final[(((gradx==255) | (grady==255)) | ((mag==255) & (dir==255)))]=255
cv2.imshow('x',gradx)
cv2.imshow('y',grady)
cv2.imshow('mag',mag)
cv2.imshow('dir',dir)
cv2.imshow('fial',final)
trial=np.dstack((gradx,grady,mag))
cv2.imshow('trial',trial)
half=np.dstack((gradx,gradx,gradx))
print(half.shape)
print(img.shape)

cv2.imshow('half',half)
cv2.waitKey()
