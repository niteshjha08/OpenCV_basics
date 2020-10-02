import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img=cv2.imread('lampard.PNG')
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(imggray,1.05,3)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('img',img)
print("Is the face detected correctly?(y/n)")
k=cv2.waitKey(0)
if(k==ord('y')):
    face=imggray[y:y+h,x:x+w]
    face=cv2.resize(face,(200,200))
    name=input("Enter the index of image:")
    cv2.imwrite('lamps_face.jpg'.format(name),face)
