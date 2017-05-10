
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


img0 = cv2.imread("..//Data//2seancephoto//rien//450.jpg", 1)
a = img0.shape
tailleImage = a[0] * a[1] * a[2]


img0 = cv2.cvtColor( img0, cv2.COLOR_RGB2GRAY )

#cv2.imshow('Display window', img0)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.show()

print("impression de la suite:")

import scipy.ndimage as nd

img0 = img0 * 1.0

out = nd.filters.gaussian_filter(img0, 8)

#cv2.imshow('Display window',out/255.);
#cv2.waitKey(0)
#cv2.destroyAllWindows()

Ex = abs(np.gradient(out)[1])

min = np.min(Ex)
max = np.max(Ex)
#Ex = 255.*(Ex-min)/(max-min)
#Ex = [[int32(s) for s in sublist] for sublist in Ex]
#Ex = np.array(Ex,dtype = np.uint8)

#cv2.waitKey(0)
#A = np.sqrt(Ex[0]**2 + Ex[1]**2)



cv2.imshow('Display window', Ex);
cv2.waitKey(0)
cv2.destroyAllWindows()

grad_h = abs(Ex)

boxNbr = 10.

import statistics as stats

stockvar = np.array([])
descripteur = [0]*len(grad_h[0])
bin = np.array([j**5 for j in np.arange(256.**(1./5.), step=(256.**(1./5.))/boxNbr)])
for i in range(len(grad_h[0])):
    #Nous avons l'histogramme et le bin.
    #L'histogramme indique le nombre de points sur la colonne qui sont assez proches en terme de couleur
    #Le bin indique l'intensite de la couleur
    #Calculons enfin la proximite geographique des valeurs au sein du bin.

    #    blabla = np.zeros((boxNbr-2,0))

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


print(bin)

#Maintenant que nous avons ce descripteur, il

#hist = [sum(grad_h[:,i]) for i in range(len(grad_h[0]))]
plt.plot(stockvar)
plt.show()


plt.plot(descripteur)
plt.show()