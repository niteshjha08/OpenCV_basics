import cv2
cap=cv2.VideoCapture(0)
_,img=cap.read()
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(cv2.waitKey(10)!=ord('q')):
    _, img = cap.read()
    faces=face_cascade.detectMultiScale(img,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('img',img)