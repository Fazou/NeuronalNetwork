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
#import os
#os.chdir("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//Imagemodifier")
import pandas as pd
import matplotlib.pyplot as plt



nombrePhoto0 = 5
nombrePhoto1 = 5
nombreReelPhoto0 = nombrePhoto0
nombreReelPhoto1 = nombrePhoto1

nombrePhoto = nombrePhoto0 + nombrePhoto1


img0 = cv2.imread("..//Data//2seancephoto//cylindrejaune//10.jpg", 1)
if(img0 != None):
    a = img0.shape
    tailleImage = a[0] * a[1] * a[2]

    #cv2.imshow('Display window', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = img0.reshape(1, tailleImage)

target = []


print("On a vu la photo maintenant on va tout importer")

for i in range(0,nombrePhoto0):
    print(i)
    image = cv2.imread("..//Data//2seancephoto//rien//"+ str(i) + ".jpg", 1)
    if (image != None):
        img0 = image.reshape(1,tailleImage)
        
        img = np.concatenate((img, img0), axis=0)
        target.append(0)
    else:
        nombreReelPhoto1 -= 1
print("on a fini les image sans rien")
for i in range(0,nombrePhoto1):
    print(i)
    image = cv2.imread("..//Data//2seancephoto//cylindrejaune//"+ str(i) + ".jpg", 1)
    if(image != None):
        img0 = image.reshape(1,tailleImage)
        X0 = reductionInput(image,100).reshape(1,100)
        #img0 = np.concatenate((img0, np.array([[0]])), axis=1)
        img = np.concatenate((img, img0), axis=0)
        X = np.concatenate((X,X0),axis = 0)
        #X.append(reductionInput(image,100))

        target.append(1)
    else:
        nombreReelPhoto1 -= 1

print(X)
nombreReelPhoto = nombreReelPhoto0 + nombreReelPhoto1




target = np.array(target)


np.savetxt("..//Data//Imagemodifier//donee.csv", X, delimiter =',')
np.savetxt("..//Data//Imagemodifier//target.csv", target, delimiter =',')
donnerecup = pd.read_csv("..//Data//Imagemodifier//donee.csv")
print(type(donnerecup))
print(donnerecup)
donnerecup = donnerecup.as_matrix()
print(donnerecup)
print("fin")

n_split = 5
kf = KFold(n_splits=n_split,shuffle = True)
n = 0
uniformResultat = [0]*5
for train, test in kf.split(donnerecup):
    print(n)
    X_k = donnerecup[train]
    Y_k = target[train]
    X_k_test = donnerecup[test]
    Y_k_test = target[test]

    #classifier = svm.SVC(gamma=0.001,kernel='rbf')
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    #classifier.fit(data[:n_samples - N], digits.target[:n_samples - N])
    classifier.fit(X_k,Y_k)
    print("On a fini l'apprentissage")
    # Now predict the value of the digit on the second half:
    Z = classifier.predict(X_k_test)
    uniformResultat[n] = sum(Z == Y_k_test) / float(len(Z))
    n = n + 1
    print(uniformResultat)