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

img = cv2.imread("..//Data//2seancephoto//cylindrejaune//0.jpg", 1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Display window', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# find Harris corners

gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,9,10**(-7))

dst = cv2.dilate(dst, None)

ret, dst = cv2.threshold(dst, 0.001 * dst.max(), 255, 0)

dst = np.uint8(dst)
cv2.imshow('Display window', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
  # find centroids

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
print(labels)
print(stats)
print(centroids)

  # define the criteria to stop and refine the corners

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

  # Now draw them

res = np.hstack((centroids, corners))

res = np.int0(res)
print(res)

img[res[:, 1], res[:, 0]] = [0, 0, 255]

img[res[:, 3], res[:, 2]] = [0, 255, 0]

cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()