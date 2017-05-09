
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
print(cv2. __version__)

for i in range(0,1500):
    print(i)
    img = cv2.imread("..//Data//2seancephoto//cylindrejaune//"+str(i)+".jpg", 1)
    #img = cv2.imread("..//Data//2seancephoto//cylindrebleu//"+str(100)+".jpg", 1)

    # Initiate STAR detector
    #orb = cv2.ORB()
    #orb = cv2.ORB(nfeatures=10, scoreType=cv2.ORB_FAST_SCORE,scaleFactor=1.9)
    orb = cv2.ORB(edgeThreshold=10, patchSize=2, nlevels=6, scaleFactor=2.3, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=200)


    #descriptor = cv2.DescriptorExtractor_create("ORB")  # BRISK
    #cv2.DescriptorExtractor_create.compute(img, keypoints)



    #a,b = orb.compute(img,kp_scene)
    #print(len(a))

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    try:
        img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)
        print(len(kp))
        #print(len(des))

        cv2.imshow('Display window', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Exitse pas"+str(i))
