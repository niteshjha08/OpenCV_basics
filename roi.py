import numpy as np
import cv2
import time
cap=cv2.VideoCapture(0)
_,img=cap.read()
time.sleep(2)
_,img=cap.read()
cv2.imshow('frame',img)
roi=img[0:230,30:220]
img[250:480,30:220]=roi
cv2.imshow('frame_new',img)
cv2.waitKey(0)