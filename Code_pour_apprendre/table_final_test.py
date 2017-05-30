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
liste_test = [i for i in range(0,200)]
#660 jaune bug !

for k in range(0,3):
    for i in liste_test:
        try:

            Img = Image.open("..//Data//table_final_tournoi//" + categorie[k] + "//" + str(i) + ".jpg")
            # Img = Image.open("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//2seancephoto//"+categorie[k]+"//" + str(i) + ".jpg")
            # Img = Img.convert('LA') #Pour noir et blanc
            Img = Img.resize((size, size), PIL.Image.ANTIALIAS)
            # Img = list(Img.getdata()) #Pour noir et blanc sinon mettre : Img = img_to_array(Img)
            Img = img_to_array(Img)
            # Img = np.array(Img)
            # Img = Img[:, 0] #Que noir et blanc !
            # Img = Img.reshape(size, size, 3) #Que noir et blanc encore !
            X_test.append(Img)


            if (k == 0):
                Y_test.append(0)
            if (k == 1):
                Y_test.append(1)
            if (k == 2):
                Y_test.append(1)
        except :
            print("Erreur : Pas d'image avec ces valeurs !",k,i)


# normalize inputs from 0-255 to 0.0-1.0

X_test = np.array(X_test)
X_test = X_test.astype('float32')
X_test = X_test / 255.0


Y_test = np_utils.to_categorical(Y_test)


#print("Size of test dataset")
#print(X_test.shape)
#print("Size of the test label")
#print(len(Y_test))

num_classes = 2
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(size, size,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 5
lrate = 0.01
decay = lrate/epochs

#load weights
model.load_weights("..//..//70weigths_color0.h5")

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

score = model.evaluate(X_test, Y_test,  verbose=0)
score2 = model.predict(X_test, verbose=0)
print(score)
print(score2[:,1])
print(Y_test[:,1])
print("Accuracy: %.2f%%" % (score[1]*100))

print("Duree de l'execution : ")
print(time.time()-start)


a = 0
IlYavaitpasdecylindre = 0
IlYavaituncylindre = 0
alpha = 0.1

for k in range(0,3):
    for i in range(len(liste_test)):
        img = cv2.imread("..//Data//table_final_tournoi/" + categorie[k] + "/" + str(liste_test[i]) + ".jpg", 1)
        if (type(img) != type(None)):
            if(score2[a][1] > alpha):
                if(Y_test[a,1]==0):
                    IlYavaitpasdecylindre = IlYavaitpasdecylindre + 1
                    cv2.putText(img,str(score2[a,1]),(10,400),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
                    print(k,i)
                    #cv2.imshow('Display window', img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                cv2.circle(img,(50,50), 50, (0,0,255), -1)
            if (score2[a][1] <= alpha):
                if (Y_test[a, 1] == 1):
                    IlYavaituncylindre = IlYavaituncylindre + 1
                    cv2.putText(img, str(score2[a, 1]), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,cv2.LINE_AA)
                    print(k,i)
                    #cv2.imshow('Display window', img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

            #cv2.putText(img,str(score2[a,1]),(10,400),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            #cv2.imshow('Display window', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            a = a + 1
            #cv2.imwrite('test_final'+str(a)+'.jpg', img)

print((IlYavaitpasdecylindre/float(a))*100)
print((IlYavaituncylindre/float(a))*100)