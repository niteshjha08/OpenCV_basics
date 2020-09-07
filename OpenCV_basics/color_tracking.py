import cv2
import time
import numpy as np


cap = cv2.VideoCapture(0)
_, img = cap.read()
cv2.imshow('frrr',img)
cv2.waitKey(0)
print('start camera')
time.sleep(2)
_, img1 = cap.read()
a=np.array([])
while (True):

    _, img = cap.read()
    img1=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grey', imgGrey)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', imghsv)
    cv2.waitKey(20)
    # light_orange=(130,40,0)
    # dark_orange =(180,255,255)
    # mask1=cv2.inRange(imghsv,light_orange,dark_orange)
    # light_red=(0,50,20)
    # dark_red =(20,255,255)
    # mask2=cv2.inRange(imghsv,light_orange,dark_orange)
    # mask=mask1+mask2
    light_blue = (75, 100, 20)
    dark_blue = (130, 255, 255)
    mask = cv2.inRange(imghsv, light_blue, dark_blue)
    M = cv2.moments(mask)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(mask, (cX, cY), 10, (0, 0, 255), 5)
    cv2.imshow('mask', mask)
    result = cv2.bitwise_and(img, img, mask=mask)
    resultgray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contour, _ = cv2.findContours(resultgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # try:
    maxcontour = max(contour, key=cv2.contourArea)
    print(maxcontour.shape)
    print(maxcontour)
    print('data:\n\n')
    print()
    cv2.waitKey(0)
   #  x, y, w, h = cv2.boundingRect(maxcontour)
   #  cv2.putText(img, "Here it is", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
   #  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
   #  cv2.drawContours(result, maxcontour, -1, (0, 255, 0), 3)
   #  cv2.imshow('Original', img)
   #  print(type(maxcontour))
   #  print(np.shape(maxcontour))
   #  print(maxcontour)
   #  hull=cv2.convexHull(maxcontour,False)
   #  cv2.drawContours(img2,[hull],-1,(0,255,0),3)
   #  cv2.imshow('hull',img2)
   #  cnt = cv2.approxPolyDP(maxcontour, 0.07 * cv2.arcLength(maxcontour, True), True)
   #  # print(np.shape(cnt))
   #  # (a,b,c)=np.shape(cnt)
   #  # cnt=np.reshape(a,2)
   #  # cnt=np.array(cnt)
   #  # print(type(cnt))
   #  # print(len(cnt))
   #  # print(np.shape(cnt))
   #
   #  #cv2.drawContours()
   #  cv2.drawContours(img1,[cnt], -1, (0, 255, 0), 3)
   # # cv2.line(img1,cnt[1],cnt[2],(0,0,255),2)
   #  cv2.imshow('approxpoly', img1)
   #  cv2.imshow('res', result)
   #  if (cv2.waitKey(1) == ord('q')):
   #      break
   #  # except:
   #  #         print("contour not found!")
   #  #         time.sleep(1)
