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
import cv2


print(cv2.__version__)

start = time.time()

#resize image
size = 70

categorie = ["rien", "cylindrejaune", "cylindrebleu"]

scores = []

Y_test = []
X_test = []
Image_test = []

#liste_test = [l for l in range(500)]
liste_test = [1,2,3,4,5,6,7,8,9]
#660 jaune bug !

for k in range(0,1):
    for i in liste_test:
        try:
            Img = Image.open("..//Data//table_final/"+categorie[k]+"/" + str(i) + ".jpg")
            Image_test.append(Img)
            #Img.save("..//Data//table_final//noirblanc//"+str(i)+"_"+str(k)+"_couleur.jpg")
            #Img.save("..//Data//table_final//noirblanc//a.jpg")

            #Img = Image.open("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//2seancephoto//"+categorie[k]+"//" + str(i) + ".jpg")
            Img = Img.convert('LA')
            Img.save("..//Data//table_final//noirblanc//"+str(i)+"_"+str(k)+"_noiretblanc.jpg")
            Img = Img.resize((size, size), PIL.Image.ANTIALIAS)
            Img.save("..//Data//table_final//noirblanc//"+str(i)+"_"+str(k)+"_noiretblanc_reduit.jpg")
            Img = list(Img.getdata())
            Img = np.array(Img)
            Img = Img[:, 0]
            Img = Img.reshape(size, size, 1)
            X_test.append(Img)


            if (k == 0):
                Y_test.append(0)
            if (k == 1):
                Y_test.append(1)
            if (k == 2):
                Y_test.append(1)
        except :
            print("Erreur : Pas d'image avec ces valeurs !",k,i)
