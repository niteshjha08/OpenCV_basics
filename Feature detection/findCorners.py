import cv2
import numpy as np

img=cv2.imread('images/chess.PNG')
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imggray=np.float32(imggray)
corners=cv2.cornerHarris(imggray,5,29,0.1)
print(corners[:10])
img[corners>0.1*corners.max()]=[0,0,255]
cv2.imshow('',img)
cv2.waitKey()
cv2.destroyAllWindows()