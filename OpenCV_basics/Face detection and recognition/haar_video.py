import cv2
cap=cv2.VideoCapture(0)
_,img=cap.read()
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
while(cv2.waitKey(10)!=ord('q')):
    _, img = cap.read()
    faces=face_cascade.detectMultiScale(img,1.3,3)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        face=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(face,1.1,3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(img, (x+ex, y+ey), (x+ex + ew,y+ ey + eh), (255, 0, 0), 0)

    cv2.imshow('window',img)