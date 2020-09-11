import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def scan(suspects_dir,tattoos_dir):
    suspects=[]
    for dir,subdir,files in os.walk(suspects_dir):
        suspects.extend(files)
    print(suspects)
    idx=input("Enter index of suspect to scan database [0-{}]:".format(len(suspects)-1))
    suspect=suspects[int(idx)]
    path=os.path.join(suspects_dir,suspect)
    img=cv2.imread(path)

    sift=cv2.xfeatures2d.SIFT_create()
    kp, d = sift.detectAndCompute(img,None)

    FLANN_INDEX_KDTREE=0
    indexParam=dict(algorithm= FLANN_INDEX_KDTREE, trees= 5)
    searchParam=dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParam,searchParam)
    MIN_MATCH_COUNT=10

    probable_suspect=[]
    files=[]
    descriptors=[]
    for dir,subdir,filename in os.walk(tattoos_dir):
        files.extend(filename)
        for f in files:
            if(f.endswith('npy')):
                descriptors.append(f)

    max_matches=0
    k=0
    for i in descriptors:
        viable_match=[]
        matches=flann.knnMatch(d,np.load(os.path.join(tattoos_dir,i)),k=2)
        for m,n in matches:
            if(m.distance<0.7*n.distance):
                viable_match.append(matches)
        if(len(viable_match))>MIN_MATCH_COUNT:
            print('{0} is a match! It had {1} viable_matches'.format(i,len(viable_match)))
        else:
            print('{0} is not a match! It had {1} viable_matches'.format(i,len(viable_match)))
        if(len(viable_match)>max_matches):
            k=i
            max_matches=len(viable_match)
        img3=cv2.drawMatches()
    print("The culprit is : {}".format(k))


scan('suspects','tattoos')