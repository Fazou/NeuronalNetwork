print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss
import time
from sklearn.model_selection import KFold
from PseudoGradient import Gradient
import pandas as pd

img1 = cv2.imread("..//Data//2seancephoto//cylindrejaune//0.jpg", 1)
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Display window', gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



gray = np.float32(gray)
list1 = [0,1,3,5,7,9,11,13,15,17,19]
list = [0,65,230,147,169,164,68,650,147,760]
#for i in range(1,2):
for i in list:
    print(i)
    img1 = cv2.imread("..//Data//2seancephoto//rien//"+str(i)+".jpg", 1)
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    try:
        dst = cv2.cornerHarris(gray,2,9,10**(-7))
    except:
        print("coucou")
    #print(dst)
    print(len(dst))
    print(len(dst[0]))
    cv2.imshow('Display window', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
#cv2.imshow('Display window', dst)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Threshold for an optimal value, it may vary depending on the image.
    print(dst.max)
    img1[dst>0.000002*dst.max()]=[0,0,255]
    #img1[dst < 0.00002 * dst.max()] = [255, 255, 255]
    # 0.00002
    a = dst.max()
    nb = 0
    for i in range(0,480):
        for j in range(0,640):
            if(dst[i][j] > 0.00002*a):
                nb = nb + 1
    print(nb)
    cv2.imshow('Display window', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
