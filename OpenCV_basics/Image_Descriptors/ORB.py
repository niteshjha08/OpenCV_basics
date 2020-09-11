import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('images/lp1.PNG',0)
img2=cv2.imread('images/lp2.PNG',0)
orb=cv2.ORB_create()
kp1,d1=orb.detectAndCompute(img,None)
kp2,d2=orb.detectAndCompute(img2,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches=bf.match(d1,d2)
matches=sorted(matches, key= lambda x: x.distance)
img3=np.array((img.shape[0],img.shape[1]))
img3=cv2.drawMatches(img,kp1,img2,kp2,matches[:40],img2,flags=2)
plt.imshow(img3,cmap="gray")
plt.show()