
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
    img = cv2.imread("..//Data//2seancephoto//cylindrebleu//"+str(i)+".jpg", 1)

    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    try:
        img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)

        cv2.imshow('Display window', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Exitse pas"+str(i))
