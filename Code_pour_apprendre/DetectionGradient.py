import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss
import time
from sklearn.model_selection import KFold
from PseudoGradient import Gradient
import operator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import np_utils
#from transform import dataset
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import KFold
import time
import scipy.ndimage as nd


start = time.time()



def kfolds(k, N, seed=None):
    random.seed(seed)
    out = [list() for _ in range(k)]
    for n in range(N):
        out[random.randrange(k)].append(n)
    return (out)


#https://github.com/fchollet/keras/issues/4465 for VGG16 model

nombrePhoto0 = 27
nombrePhoto1 = 20
nombrePhoto2 = 20

nombrePhoto = [nombrePhoto0,nombrePhoto1,nombrePhoto2]


resultat=[]
resultatParam = []
resultatFauxPositif=[]
resultatFauxNegatif=[]
categorie = ["rien", "cylindrejaune", "cylindrebleu"]

n_split = 5

K = kfolds(seed=486684,k=n_split,N=max(nombrePhoto))
uniformResultat = [0] * n_split
uniformResultatFauxPositif = [0] * n_split
uniformResultatFauxNegatif = [0] * n_split
scores = []

for n in range(0, n_split):
    Y_train = []
    X_train = []
    Y_test = []
    X_test = []
    print(n)

    list1 = [i for i in range(0, n_split)]
    list1.remove(n)
    train = []
    for j in list1:
        train = sum(K[j:j + 1], train)
    test = K[n]

    for k in range(0, 3):
        print(k)
        for i in train:
            try:
                print(i, k,"train")

                img0 = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+str(i)+".jpg", 1)
                #print(type(img0)==type(None))

                img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)

                #cv2.imshow('Display window', img0)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                img0 = img0 * 1.0

                out = nd.filters.gaussian_filter(img0, 8)
                Ex = abs(np.gradient(out)[1])

                min = np.min(Ex)
                max = np.max(Ex)

                grad_h = abs(Ex)

                #cv2.imshow('Display window', grad_h)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                boxNbr = 10.  # nbre de cluster !

                stockvar = np.array([])
                descripteur = [0] * len(grad_h[0])
                bin = np.array([j ** 5 for j in np.arange(256. ** (1. / 5.), step=(256. ** (1. / 5.)) / boxNbr)])


                best_valeur = [0] * len(grad_h[0])
                med_valeur = [0] * len(grad_h[0])

                for j in range(len(grad_h[0])):
                    nbre_point_suite = 0
                    nbre_point_suite_max = 0
                    for l in range(len(bin) - 1):
                        if (bin[l] <= grad_h[0, j] < bin[l + 1]):
                            valeur_point = l
                    valeur_point_max = 0

                    for i in range(1, len(grad_h)):
                        if (grad_h[i, j] >= bin[valeur_point - 1] and grad_h[i, j] < bin[valeur_point + 1]):
                            nbre_point_suite += 1
                        else:
                            for l in range(len(bin) - 1):
                                if (bin[l] <= grad_h[i, j] < bin[l + 1]):
                                    valeur_point = l
                            nbre_point_suite = 0
                        if (nbre_point_suite_max < nbre_point_suite):
                            nbre_point_suite_max = nbre_point_suite
                            valeur_point_max = valeur_point
                    med_valeur[j] = nbre_point_suite_max
                    best_valeur[j] = (bin[valeur_point_max] + bin[valeur_point_max]) / 2.
                    descripteur[j] = med_valeur[j] * med_valeur[j] * best_valeur[j]
                X_train.append(np.array(descripteur))
                #plt.plot(descripteur)
                #plt.show()

                if (k == 0):
                    Y_train.append(0)
                if (k == 1):
                    Y_train.append(1)
                if (k == 2):
                    Y_train.append(1)
            except:
                blabla = 1
                print("erreur ?")



        for i in test:
            try:
                print(i, k,"test")

                img0 = cv2.imread("..//Data//2seancephoto//"+categorie[k]+"//"+str(i)+".jpg", 1)
                #print(type(img0)==type(None))
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)

                #cv2.imshow('Display window', img0)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                img0 = img0 * 1.0

                out = nd.filters.gaussian_filter(img0, 8)

                Ex = abs(np.gradient(out)[1])

                #cv2.imshow('Display window', grad_h)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                min = np.min(Ex)
                max = np.max(Ex)

                grad_h = abs(Ex)



                boxNbr = 10.  # nbre de cluster !

                stockvar = np.array([])
                descripteur = [0] * len(grad_h[0])
                bin = np.array([j ** 5 for j in np.arange(256. ** (1. / 5.), step=(256. ** (1. / 5.)) / boxNbr)])

                best_valeur = [0] * len(grad_h[0])
                med_valeur = [0] * len(grad_h[0])

                for j in range(len(grad_h[0])):
                    nbre_point_suite = 0
                    nbre_point_suite_max = 0
                    for l in range(len(bin) - 1):
                        if (bin[l] <= grad_h[0, j] < bin[l + 1]):
                            valeur_point = l
                    valeur_point_max = 0

                    for i in range(1, len(grad_h)):
                        if (grad_h[i, j] >= bin[valeur_point - 1] and grad_h[i, j] < bin[valeur_point + 1]):
                            nbre_point_suite += 1
                        else:
                            for l in range(len(bin) - 1):
                                if (bin[l] <= grad_h[i, j] < bin[l+ 1]):
                                    valeur_point = l
                            nbre_point_suite = 0
                        if (nbre_point_suite_max < nbre_point_suite):
                            nbre_point_suite_max = nbre_point_suite
                            valeur_point_max = valeur_point
                    med_valeur[j] = nbre_point_suite_max
                    best_valeur[j] = (bin[valeur_point_max] + bin[valeur_point_max]) / 2.
                    descripteur[j] = med_valeur[j] * med_valeur[j] * best_valeur[j]
                X_test.append(np.array(descripteur))
                #plt.plot(descripteur)
                #plt.show()

                if (k == 0):
                    Y_test.append(0)
                if (k == 1):
                    Y_test.append(1)
                if (k == 2):
                    Y_test.append(1)
            except:
                blabla = 1
                print("erreur ?")

    print("On a creer X")

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    X_train = X_train/np.amax(X_train)
    #X_test = X_train/np.amax(X_train)


    print(Y_test)
    print(Y_train)
    print(X_test)
    print(X_train)


    #classifier = svm.SVC(gamma=0.001,kernel='rbf')
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(400, 100,20,5), random_state=1)
    #(90, 44, 5)
    #classifier.fit(data[:n_samples - N], digits.target[:n_samples - N])
    classifier.fit(X_train,Y_train)
    print("On a fini l'apprentissage")
    # Now predict the value of the digit on the second half:
    Z = classifier.predict(X_test)
    print(Z)
    print(type(Z))
    print(type(Y_test))
    #print(Y_k_test)
    print(sum(Z == Y_test))
    uniformResultat[n] = sum(Z == Y_test)
    uniformResultat[n] = uniformResultat[n]/float(len(Z))
    for l in range(len(Z)):
        if(Y_test[l]==0):
            uniformResultatFauxPositif[n] += (Z[l]==1)
            # il n y a pas de cylindre mais le programme en detecte un
        if(Y_test[l]==1):
            uniformResultatFauxNegatif[n] += (Z[l]==0)
            # il y a un cylindre mais le programme n en detecte pas

    uniformResultatFauxPositif[n] = uniformResultatFauxPositif[n]/float(len(Z))
    uniformResultatFauxNegatif[n] = uniformResultatFauxNegatif[n]/float(len(Z))

    print(uniformResultat,uniformResultatFauxNegatif,uniformResultatFauxPositif)
    end = time.time()
    print(Z)
    print(Y_test)
    print(end - start)

resultat.append(sum(uniformResultat)/5)
resultatFauxPositif.append(sum(uniformResultatFauxPositif)/5)
resultatFauxNegatif.append(sum(uniformResultatFauxNegatif)/5)
print(resultat,resultatFauxNegatif,resultatFauxPositif)

print(time.time()-start)

print(scores)





""""
for i in range(len(grad_h[0])):

    blabla = np.zeros((int(boxNbr)-2,0))

    var = np.array([])
    med = np.array([])
    for k in range(1,len(bin)-1):
        elementsOfBin = [l for l in range(len(grad_h[:,i])) if grad_h[l,i] >= bin[k-1] and grad_h[l,i] < bin[k+1]]
        if(len(elementsOfBin) > 10):
            med = np.append(med,stats.median([grad_h[p,i] for p in elementsOfBin]))
            #if(i==280):
            #    print(med)
            sorted = np.sort(elementsOfBin)
            best = 0.
            current = 0.
            prec = sorted[0]
            for m in sorted[1:]:
                if(m == prec + 1.):
                    current = current + 1.
                else:
                    #best = np.max((best,current))
                    current = 0.
                best = np.max((best, current))
                #if(i==280):
                #    print(best)
                prec = m
        else:
            best = 0
            med = np.append(med, 0)
        var = np.append(var,best)

    descripteur[i] = (med*var*var).max()
    stockvar = np.append(stockvar,var.max())
    if(i==280):
        print(var)
        print(med*var)

"""

#Maintenant que nous avons ce descripteur, il

#hist = [sum(grad_h[:,i]) for i in range(len(grad_h[0]))]
#plt.plot(stockvar)
#plt.show()


