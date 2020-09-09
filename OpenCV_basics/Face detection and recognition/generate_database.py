import cv2
cap=cv2.VideoCapture(0)
_,img=cap.read()
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
index=42
while(cv2.waitKey(10)!=ord('q')):
    _, img = cap.read()
    faces=face_cascade.detectMultiScale(img,1.3,3)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        frame=cv2.resize(img[y:y+h,x:x+w],(200,200))
        cv2.imwrite('newfaces/img{}.pgm'.format(index),frame)
        index=index+1

    cv2.imshow('window',img)