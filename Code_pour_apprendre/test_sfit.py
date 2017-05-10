import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss
import time
from sklearn.model_selection import KFold
from PseudoGradient import Gradient
# import os
# os.chdir("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//Imagemodifier")
import pandas as pd
from sklearn import mixture
import random


def kfolds(k, N, seed=None):
    random.seed(seed)
    out = [list() for _ in range(k)]
    for n in range(N):
        out[random.randrange(k)].append(n)
    return (out)


nombrePhoto0 = 1700
nombrePhoto1 = 1000
nombrePhoto2 = 1000

nombrePhoto = [nombrePhoto0, nombrePhoto1, nombrePhoto2]
nombreReelPhoto = nombrePhoto




histo = pd.read_csv("..//Data//Imagemodifier//donee.csv")
target = pd.read_csv("..//Data//Imagemodifier//target.csv")
histo_test = pd.read_csv("..//Data//Imagemodifier//donee_test.csv")
target_test = pd.read_csv("..//Data//Imagemodifier//target_test.csv")

histo = histo.as_matrix()
target = target.as_matrix()
histo_test = histo_test.as_matrix()
target_test = target_test.as_matrix()

Y_total = np.concatenate((target,target_test),axis=0)
X_total = np.concatenate((histo,histo_test),axis=0)

Y_total = Y_total.reshape(1,-1)[0]
target = target.reshape(1, -1)[0]
target_test = target_test.reshape(1, -1)[0]

resultat = []
resultatParam = []
resultatFauxPositif = []
resultatFauxNegatif = []

best_i = 0
best_j = 0
best = 0
#0.012
param1 = [200]
param2 = [110]
uniformResultat = [0]*len(param1)

for i in range(0,len(param1)):
    uniformResultat[i]=[i]*(len(param2))
    for j in range(0,len(param2)):
        print(param1[i],param2[j])
        K = kfolds(seed=4863452, k=5, N=len(X_total))

        n_split = 5
        kf = KFold(n_splits=n_split, shuffle=True)
        uniformResultat_temporaire = [0] * n_split
        uniformResultatFauxPositif = [0] * n_split
        uniformResultatFauxNegatif = [0] * n_split

        for n in range(0, n_split):
            #print(n)
            list1 = [b for b in range(0, n_split)]
            list1.remove(n)
            train = []
            for k in list1:
                train = sum(K[k:k + 1], train)
            test = K[n]
            X_k = X_total[train]
            Y_k = Y_total[train]
            X_k_test = X_total[test]
            Y_k_test = Y_total[test]

            #classifier = svm.SVC(gamma=0.001*param1[i],kernel='rbf',degree=2)
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(param1[i],param2[j],40), random_state=1)

            classifier.fit(X_k,Y_k)
            Z = classifier.predict(X_k_test)
            """for pred in range(len(Z)):
                if(Z[pred]==Y_k_test[pred]):
                    print(pred,Y_k_test[pred])"""
            uniformResultat_temporaire[n] = (sum(Z == Y_k_test) / float(len(Z)))
            #print(uniformResultat_temporaire)
        uniformResultat[i][j] = (sum(uniformResultat_temporaire)/float(len(uniformResultat_temporaire)))
        print(uniformResultat[i][j])
        if(best<uniformResultat[i][j]):
            best = uniformResultat[i][j]
            best_i = param1[i]
            best_j = param2[j]
            #90;44;5
            #200,110,40

#
#(0.95194583515613085, 200, 110,40)
#(0.950, 200, 110,40,10)




print(uniformResultat)
print(best,best_i,best_j)