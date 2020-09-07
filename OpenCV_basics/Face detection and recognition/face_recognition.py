import os
import cv2
import csv
import numpy as np


def read_images(index_path,imgdb_dir):
    imgarray,imglabel=[],[]
    with open(index_path) as f:
        data=csv.reader(f)
        os.chdir(os.path.join(os.getcwd(), imgdb_dir))
        for row in data:
             im=cv2.imread(row[0]+'.PGM')
             imgarray.append(np.asarray(im).astype('uint8'))
             imglabel.append(int(row[1]))
    return (imgarray,imglabel)

def recognize(X,y):
    cap=cv2.VideoCapture(0)
    print(y)
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X),np.asarray(y))
    names=['Nitesh']
    path='D:/My Github/OpenCV_basics/Face detection and recognition/'
    os.chdir(path)
    print(os.getcwd())
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while(cv2.waitKey(10)!=ord('q')):
        _,img=cap.read()
        imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(imggray,1.05,3)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            face=imggray[y:y+h,x:x+w]
            roi=cv2.resize(face,(200,200),interpolation=cv2.INTER_LINEAR)
            result=model.predict(roi)
            print('label {0}, confidence:{1}.'.format(result[0],result[1]))
            cv2.putText(img,names[result[0]],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(img, (str(100.0-result[1])+"%"), (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame',img)

if __name__=="__main__":
    index_file_path=os.path.join(os.curdir,'file_index.csv')
    print(index_file_path)
    imgdb_dir='faces_db'
    imgarray,imglabel=read_images(index_file_path,imgdb_dir)
    recognize(imgarray,imglabel)
