import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('images/lp1.PNG',0)
img2=cv2.imread('images/lp2.PNG',0)

sift=cv2.xfeatures2d.SIFT_create()
kp1,d1=sift.detectAndCompute(img,None)
kp2,d2=sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE=0
indexParam=dict(algorithm= FLANN_INDEX_KDTREE,trees=5)
searchParam=dict(checks=50)

flann=cv2.FlannBasedMatcher(indexParam,searchParam)
matches=flann.knnMatch(d1,d2,k=2)
matchesMask=[[0,0] for i in range(len(matches))]
for i, (m,n) in enumerate(matches):
    if(m.distance)<0.7*n.distance:
        matchesMask[i]=[1,0]
drawParams = dict(matchColor = (0,255,0),singlePointColor = (0,0,255),matchesMask = matchesMask,flags = 0)
resultImage =cv2.drawMatchesKnn(img,kp1,img2,kp2,matches,None,**drawParams)
plt.imshow(resultImage,), plt.show()

