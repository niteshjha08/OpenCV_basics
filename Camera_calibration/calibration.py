import cv2
import numpy as np
import os
nx=8
ny=6
image_list=[]
for dir,subdir,files in os.walk('.'):
    for filename in files:
            if(filename.endswith('.jpg')):
                image_list.append(os.path.join(dir,filename))


objectpoints=[]
imagepoints=[]
objp=np.zeros((48,3),dtype=np.float32)
objp[:,:2]=np.mgrid[:8,:6].T.reshape(-1,2)
for imagepath in image_list:
    img=cv2.imread(imagepath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(nx,ny),None)
    if(ret==True):
        imagepoints.append(corners)
        objectpoints.append(objp)

ret,mtx,dist,rvects,tvecs=cv2.calibrateCamera(objectpoints,imagepoints,gray.shape[::-1],None,None)
img=cv2.imread('calibration_test.png')
dst=cv2.undistort(img,mtx,dist,None,mtx)
print('image size is',img.shape)
cv2.imshow('frame',dst)
print('dst size is',dst.shape)
cv2.imwrite('saved_new.jpg',dst)
cv2.waitKey()
