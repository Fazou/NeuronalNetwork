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

#Fonction qui prend en argument une arraylist image et qui rend un arraylist plus petit
# on doit donc avoir une liste de liste
# taille_vecteur est la taille final de notre vecteur
def reductionInput(img,taille_vecteur):
    print(img)
    racine = int(np.sqrt(taille_vecteur))
    nbrel = len(img)/racine
    print()
    nbrec = len(img[0])/racine
    print(nbrel)
    print(nbrec)
    petit_array = [0]*taille_vecteur
    for i in range(0,racine):#ligne
        for j in range(0,racine):#colonne
            print((j+1)*nbrec)
            moyenne = 0
            for k in range(i*nbrel,(i+1)*nbrel):
                for l in range(j*nbrec,(j+1)*nbrec):
                    if(k == 455):
                        print(k,l)
                    if(img[k][l][1] < 150 and img[k][l][1] > 50 and img[k][l][2] < 150 and img[k][l][2] > 50 and img[k][l][0] < 20 and img[k][l][0] > 0):# [0, 50, 50], [20, 150, 150])
                        moyenne += 1
                        #print(moyenne)
            #print(moyenne/(nbrel*nbrec))
            petit_array[10*i+j] = moyenne
    print(np.array(petit_array))
    return(np.array(petit_array))



nombrePhoto0 = 180
nombrePhoto1 = 180

nombrePhoto = nombrePhoto0 + nombrePhoto1


img0 = cv2.imread("..//Data//2seancephoto//cylindrejaune//0.jpg", 1)
if(img0 != None):
    a = img0.shape
    tailleImage = a[0] * a[1] * a[2]
    cv2.imshow('Display window', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Debut de reduction input")
reductionInput(img0,100)
print("Fin de reduction input")

img = img0.reshape(1, tailleImage)

#img = np.concatenate((img,np.array([[0]])),axis=1)

target = [0]


print("On a vu la photo maintenant on va tout importer")

for i in range(0,nombrePhoto0):
    image = cv2.imread("..//Data//Label0//"+ str(i) + ".jpg", 1)
    if (image != None):
        img0 = image.reshape(1,tailleImage)
        #img0 = np.concatenate((img0, np.array([[0]])), axis=1)
        img = np.concatenate((img, img0), axis=0)
        target.append(0)
    else:
        nombrePhoto -= 1
        nombrePhoto0 -= 1
for i in range(1,nombrePhoto1):
    image = cv2.imread("..//Data//Label1//"+ str(i) + ".jpg", 1)
    if(image != None):
        img0 = image.reshape(1,tailleImage)
        #img0 = np.concatenate((img0, np.array([[1]])), axis=1)
        img = np.concatenate((img, img0), axis=0)
        target.append(1)
    else:
        nombrePhoto -= 1
        nombrePhoto1 -= 1


target = np.array(target)
print(img)

kf = KFold(n_splits=5)
for train, test in kf.split(img):
    print("%s %s" % (train, test))



#classifier = svm.SVC(gamma=0.001,kernel='rbf')
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

#classifier.fit(data[:n_samples - N], digits.target[:n_samples - N])
classifier.fit(img,target)
print("On a fini l'apprentissage")
# Now predict the value of the digit on the second half:
expected = target
predicted = img

print(target)
Z = classifier.predict(img)
print(Z)
uniformResultat = sum(Z == target) / float(len(Z))

print(uniformResultat)