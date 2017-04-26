
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss
import time
from sklearn.model_selection import KFold
from PseudoGradient import Gradient





nombrePhoto0 = 30
nombrePhoto1 = 30

nombrePhoto = nombrePhoto0 + nombrePhoto1


img0 = cv2.imread("..//Data//2seancephoto//cylindrejaune//1.jpg", 1)
a = img0.shape
tailleImage = a[0] * a[1] * a[2]


img0 = cv2.cvtColor( img0, cv2.COLOR_RGB2GRAY )

cv2.imshow('Display window', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("impression de la suite:")
Ex= np.gradient(img0)
print(len(Ex[0][0]))
print(img0)
A = np.sqrt(Ex[0]**2 + Ex[1]**2)

cv2.imshow('Display window', Ex[0]);
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Display window', Ex[1]);
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Display window', A);
cv2.waitKey(0)
cv2.destroyAllWindows()