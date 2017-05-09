import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import mixture
import random
import time

start = time.time()

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
resultatParam = []
resultatFauxPositif=[]
resultatFauxNegatif=[]
categorie = ["rien", "cylindrejaune", "cylindrebleu"]

#x,y=0,0
#w,h=640-1,480-1
x,y=0+75,25
w,h=680-125-1,480-25-1

for nbre_cluster in range(2,3):

    K = kfolds(seed=4863452,k=5,N=max(nombrePhoto))

    n_split = 5
    kf = KFold(n_splits=n_split,shuffle = True)
    uniformResultat = [0]*n_split
    uniformResultatFauxPositif = [0]*n_split
    uniformResultatFauxNegatif = [0]*n_split



    for n in range(0,n_split):
        print(n)
        #detector = cv2.FeatureDetector_create("ORB")  # ORB
        #descriptor = cv2.DescriptorExtractor_create("ORB")  # BRISK
        #orb = cv2.ORB(edgeThreshold=10, patchSize=2, nlevels=6, scaleFactor=2.3, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=200)
        orb =  cv2.ORB()
        X=[]

        list1 = [i for i in range(0,n_split)]
        list1.remove(n)
        train = []
        for j in list1:
            train = sum(K[j:j + 1], train)
        test = K[n]


        for k in range (0,3):
            for i in train:
                image = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+ str(i) + ".jpg", 1)
                if (type(image) != type(None)):
                    #image = image[y:h, x:w]  # Crop from x, y, w, h -> 100, 200, 300, 400
                    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

                    #kp_scene = detector.detect(image)
                    kp_scene = orb.detect(image)


                    #k_scene, d_scene = descriptor.compute(image, kp_scene)
                    k_scene, d_scene = orb.compute(image, kp_scene)

                    try:
                        for j in range(0, len(d_scene)):
                            X.append(d_scene[j])
                    except:
                        #print(str(i) + " ; " + str(k) + " plante du poney")
                        ae = 4

        print("On a creer X")

        gmm = mixture.GaussianMixture(n_components=nbre_cluster*1, covariance_type='full').fit(X)

        print("fini l'apprentissage K-means")

        histo = []
        target = []
        target_test = []
        histo_test = []
        for k in range (0,3):
            for i in train:
                image = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+ str(i) + ".jpg", 1)

                if (type(image) != type(None)):
                    #image = image[y:h, x:w]  # Crop from x, y, w, h -> 100, 200, 300, 400

                    #kp_scene = detector.detect(image)
                    #k_scene, d_scene = descriptor.compute(image, kp_scene)
                    kp_scene = orb.detect(image)
                    k_scene, d_scene = orb.compute(image, kp_scene)
                    try:
                        a = sum(gmm.predict_proba(d_scene))
                        histo.append(a)
                        if (k == 0):
                            target.append(0)
                        if (k == 1):
                            target.append(1)
                        if (k == 2):
                            target.append(1)
                    except:
                        print(str(i) + " ; " + str(k) + " bug")

                else:
                    nombreReelPhoto[k] -= 1
            for i in test:
                image = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+ str(i) + ".jpg", 1)

                if (type(image) != type(None)):
                    #image = image[y:h, x:w]  # Crop from x, y, w, h -> 100, 200, 300, 400

                    #kp_scene = detector.detect(image)
                    #k_scene, d_scene = descriptor.compute(image, kp_scene)
                    kp_scene = orb.detect(image)
                    k_scene, d_scene = orb.compute(image, kp_scene)

                    try:
                        a = sum(gmm.predict_proba(d_scene))
                        histo_test.append(a)
                        if (k == 0):
                            target_test.append(0)
                        if(k == 1):
                            target_test.append(1)
                        if (k == 2):
                            target_test.append(1)
                    except:
                        print(str(i) +" ; " + str(k)+" bug")
                        ae = 4
                else:
                    nombreReelPhoto[k] -= 1

        np.savetxt("..//Data//Imagemodifier//donee"+str(n)+".csv", histo, delimiter =',')
        np.savetxt("..//Data//Imagemodifier//donee_test"+str(n)+".csv", histo_test, delimiter =',')
        np.savetxt("..//Data//Imagemodifier//target"+str(n)+".csv", target, delimiter =',')
        np.savetxt("..//Data//Imagemodifier//target_test"+str(n)+".csv", target_test, delimiter =',')

        histo = pd.read_csv("..//Data//Imagemodifier//donee"+str(n)+".csv")
        target = pd.read_csv("..//Data//Imagemodifier//target"+str(n)+".csv")
        histo_test = pd.read_csv("..//Data//Imagemodifier//donee_test"+str(n)+".csv")
        target_test = pd.read_csv("..//Data//Imagemodifier//target_test"+str(n)+".csv")

        histo = histo.as_matrix()
        target = target.as_matrix()
        histo_test = histo_test.as_matrix()
        target_test = target_test.as_matrix()
        target= target.reshape(1,-1)[0]
        target_test= target_test.reshape(1,-1)[0]

        X_k = histo
        Y_k = target
        X_k_test = histo_test
        Y_k_test = target_test

        #classifier = svm.SVC(gamma=0.001,kernel='rbf')
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 4), random_state=1)
        #(90, 44, 5)
        #classifier.fit(data[:n_samples - N], digits.target[:n_samples - N])
        classifier.fit(X_k,Y_k)
        print("On a fini l'apprentissage")
        # Now predict the value of the digit on the second half:
        Z = classifier.predict(X_k_test)
        #print(Z)
        #print(Y_k_test)

        uniformResultat[n] = sum(Z == Y_k_test) / float(len(Z))
        for l in range(len(Z)):
            if(Y_k_test[l]==0):
                uniformResultatFauxPositif[n] += (Z[l]==1)
                # il n y a pas de cylindre mais le programme en detecte un
            if(Y_k_test[l]==1):
                uniformResultatFauxNegatif[n] += (Z[l]==0)
                # il y a un cylindre mais le programme n en detecte pas

        uniformResultatFauxPositif[n] = uniformResultatFauxPositif[n]/float(len(Z))
        uniformResultatFauxNegatif[n] = uniformResultatFauxNegatif[n]/float(len(Z))

        print(uniformResultat,uniformResultatFauxNegatif,uniformResultatFauxPositif)
        end = time.time()
        print(end - start)

    resultat.append(sum(uniformResultat)/5)
    resultatFauxPositif.append(sum(uniformResultatFauxPositif)/5)
    resultatFauxNegatif.append(sum(uniformResultatFauxNegatif)/5)
    resultatParam.append(nbre_cluster*1)
    print(resultat,resultatParam,resultatFauxNegatif,resultatFauxPositif)
