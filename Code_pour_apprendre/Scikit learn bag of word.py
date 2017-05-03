
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
from sklearn import mixture
import random
from sklearn.cluster import KMeans

def kfolds(k,N,seed=None):
    random.seed(seed)
    out = [ list() for _ in range(k) ]
    for n in range(N):
        out[random.randrange(k)].append(n)
    return(out)


nombrePhoto0 = 1700
nombrePhoto1 = 1000
nombrePhoto2 = 1000

nombrePhoto = [nombrePhoto0,nombrePhoto1,nombrePhoto2]
nombreReelPhoto = nombrePhoto

resultat=[]



for nbre_cluster in range(1,10):

    K = kfolds(seed=3894,k=5,N=max(nombrePhoto))

    n_split = 5
    kf = KFold(n_splits=n_split,shuffle = True)
    uniformResultat = [0]*n_split


    for n in range(0,n_split):
        nbre_erreur = 0
        print(n)
        detector = cv2.FeatureDetector_create("ORB")  # SURF
        descriptor = cv2.DescriptorExtractor_create("ORB")  # BRIEF
        X=[]

        train = []
        list1 = [i for i in range(0,n_split)]
        list1.remove(n)
        for j in list1:
            train = sum(K[j:j + 1], train)
        test = K[n]

        categorie = ["rien","cylindrejaune","cylindrebleu"]
        for k in range (0,3):
            for i in train:
                image = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+ str(i) + ".jpg", 1)
                if (type(image) != type(None)):

                    kp_scene = detector.detect(image)
                    k_scene, d_scene = descriptor.compute(image, kp_scene)
                    try:
                        for j in range(0, len(d_scene)):
                            X.append(d_scene[j])
                    except:
                        #print(str(i) + " ; " + str(k) + " plante du poney")
                        ae = 4
                        #print("erreur")

        print("On a creer X")

        kmeans = KMeans(n_clusters=nbre_cluster*100, random_state=4854964,n_jobs=-1).fit(X)

        print("fini l'apprentissage K-means")

        histo = []
        target = []
        target_test = []
        histo_test = []
        for k in range (0,3):
            for i in train:
                image = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+ str(i) + ".jpg", 1)
                if (type(image) != type(None)):
                    kp_scene = detector.detect(image)
                    k_scene, d_scene = descriptor.compute(image, kp_scene)
                    try:
                        a = kmeans.predict(d_scene)
                        somme = []
                        for nbre in range(0, nbre_cluster):
                            somme.append(len(a[a == nbre]))
                        histo.append(somme)
                        if (k == 0):
                            target.append(0)
                        if (k == 1):
                            target.append(1)
                        if (k == 2):
                            target.append(1)
                    except:
                        #print(str(i) + " ; " + str(k) + " plante du poney")
                        ae = 4
                        #print("erreur")
                        nbre_erreur += 1

                else:
                    nombreReelPhoto[k] -= 1
            for i in test:
                image = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+ str(i) + ".jpg", 1)
                if (type(image) != type(None)):
                    kp_scene = detector.detect(image)
                    k_scene, d_scene = descriptor.compute(image, kp_scene)
                    try:
                        a = kmeans.predict(d_scene)
                        somme = []
                        for nbre in range(0,nbre_cluster):
                            somme.append(len(a[a==nbre]))
                        histo_test.append(np.array(somme))
                        if (k == 0):
                            target_test.append(0)
                        if(k == 1):
                            target_test.append(1)
                        if (k == 2):
                            target_test.append(1)
                    except:
                        #print(str(i) +" ; " + str(k)+" plante du poney")
                        ae = 4
                        nbre_erreur += 1
                else:
                    nombreReelPhoto[k] -= 1

        np.savetxt("..//Data//Imagemodifier//donee.csv", histo, delimiter =',')
        np.savetxt("..//Data//Imagemodifier//target.csv", target, delimiter =',')
        histo = pd.read_csv("..//Data//Imagemodifier//donee.csv")
        target = pd.read_csv("..//Data//Imagemodifier//target.csv")

        histo = histo.as_matrix()
        target = target.as_matrix()
        target= target.reshape(1,-1)[0]

        X_k = histo
        Y_k = target
        X_k_test = histo_test
        Y_k_test = target_test

        #classifier = svm.SVC(gamma=0.001,kernel='rbf')
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        #classifier.fit(data[:n_samples - N], digits.target[:n_samples - N])
        classifier.fit(X_k,Y_k)
        print("On a fini l'apprentissage")
        # Now predict the value of the digit on the second half:
        Z = classifier.predict(X_k_test)
        #print(Z)
        #print(Y_k_test)
        uniformResultat[n] = sum(Z == Y_k_test) / float(len(Z))
        print(uniformResultat)
        print("nbre d'erreur : " + str(nbre_erreur))

    resultat.append(sum(uniformResultat)/5)
    resultat.append(nbre_cluster*100)
    print(resultat)