import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img=mpimg.imread('trafficsign.png')

src=np.array([[111,69],[310,123],[120,222],[318,247]],dtype=np.float32)

dst=np.array([[111,69],[380,69],[111,219],[380,219]],dtype=np.float32)
M=cv2.getPerspectiveTransform(src,dst)
warped=cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(warped)
cv2.Sobel(gray,)
plt.show()