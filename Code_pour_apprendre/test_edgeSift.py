
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
print(cv2. __version__)
import sys
#print (sys.version)
#print(cv2.version)
img = cv2.imread("..//Data//2seancephoto//cylindrejaune//0.jpg", 1)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")
print(fea_det)
print(des_ext)
kpts = fea_det.detect(img)
kpts, des = des_ext.compute(img, kpts)
des_list = []

for i in (0,1):
    print(i)
    img = cv2.imread("..//Data//2seancephoto//cylindrejaune//"+str(i)+".jpg", 1)
    cv2.imshow('Display window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kpts = fea_det.detect(img)
    print(i)
    kpts, des = des_ext.compute(img, kpts)
    des_list.append((i, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for i, descriptor in des_list[1:]:
    print(i)
    descriptors = np.vstack((descriptors, descriptor))

k = 100
voc, variance = kmeans(descriptors, k, 1)
#sift = cv2.xfeatures2d.SIFT_create()
#detector = cv2.SURF()
#sift = cv2.SIFT()
#kp = sift.detect(gray,None)

#img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)



