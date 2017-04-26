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

img1 = cv2.imread("..//Data//2seancephoto//cylindrejaune//280.jpg", 1)
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
cv2.imshow('Display window', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(1,10):
    corners = cv2.goodFeaturesToTrack(gray,100,10**(-8),i*10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img1,(x,y),3,255,-1)

    cv2.imshow('Display window', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
