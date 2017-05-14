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
# Fonction qui permet d'enregistrer les images qui nous servirons a l'apprentissage

# Import des modules necessaires
import numpy as np
import cv2
import sys
import time
start = time.time()


# Defini le nombre de photo prise
nbrephoto = 0

# Defini quel est le label des photos prises
label = 1

# On fixe le nombre de pixel en hauteur et en largeur
pixelHauteur = 480
pixelLargeur = 640

# Valeur pour rogner l'image
premierPixelHauteur = 0
dernierPixelHauteur = 480
premierPixelLargeur = 0
dernierPixelLargeur = 640

# Boolean qui defini si la personne a appuyer sur q
stop = True


# Un tour de boucle prend une photo, on arrete quand l utilisateur a appuyer sur "q"
while(stop):

    # Permet de lire sur la webcam
    cap = cv2.VideoCapture(0)

    # On defini le type de la video pour enregistrer
    fourcc = cv2.cv.CV_FOURCC('i', 'Y', 'U', 'V')

    # On recupere la video
    #out = cv2.VideoWriter('videoEnregistrer.avi', fourcc, 20.0, (640, 480))

    # Lit la video
    ret, frame = cap.read()

    # On quitte le programme si on arrive pas a lire
    if ret:

        # On met l'image en gris
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # On affiche l'image et on attend que l'utilisateur indique qu'ils faut passer a la prise suivante
        while True:
            ret, frame = cap.read()

            #On passe en noir et blanc
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # On redefini la taille pour enregistrer les images qui nous seront utiles
            #frame = cv2.resize(frame, (pixelLargeur, pixelHauteur))

            # Permet de rogner l'image
            #frame = frame[premierPixelLargeur:dernierPixelLargeur, premierPixelHauteur:dernierPixelHauteur]

            cv2.imshow('image', frame) #On affiche l image
            a = cv2.waitKey(1) # si l'utilisateur appuye sur une touche, on stock le resultat dans a
            #print(a)
            if(a==1048691): # 115 correspond a la lettre q
                break
            if(a==1048689): # 113 correspond a la lettre s
                stop = False
                break

        # On redefini la taille pour enregistrer les images qui nous seront utiles
        frame = cv2.resize(frame, (pixelLargeur, pixelHauteur))

        frame = frame[premierPixelHauteur:dernierPixelHauteur, premierPixelLargeur:dernierPixelLargeur]

        # On enregistre l'image dans le bon dossier AMarche que sous linux
        #cv2.imwrite("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//Label"+str(label)+"//"+str(nbrephoto)+".jpg", frame);
        cv2.imwrite("..//..//Data//VraiPhoto//"+str(nbrephoto)+".jpg", frame)

        # On rajoute une photo donc on incremente
        nbrephoto += 1
        print 'nombre de photo prise : ', nbrephoto

    else:
        break

    # On relache tout
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
print ('nombre de photo final prise : '+ str(nbrephoto))



#resize image
size = 70

categorie = ["rien", "cylindrejaune", "cylindrebleu"]

scores = []

Y_test = []
X_test = []
Image_test = []

liste_test = [i for i in range(nbrephoto)]
#660 jaune bug !

for k in range(0, 2):
    for i in liste_test:
        try:
            Img = Image.open("..//..//Data//VraiPhoto//" + str(i) + ".jpg")
            Image_test.append(Img)

            #Img = Image.open("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//2seancephoto//"+categorie[k]+"//" + str(i) + ".jpg")
            Img = Img.convert('LA')
            Img = Img.resize((size, size), PIL.Image.ANTIALIAS)

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
        except:
            print("Erreur : Pas d'image avec ces valeurs !")


# normalize inputs from 0-255 to 0.0-1.0

X_test = np.array(X_test)
X_test = X_test.astype('float32')
X_test = X_test / 255.0

print(Y_test)
Y_test = np_utils.to_categorical(Y_test)
print(Y_test)
#print("Size of test dataset")
#print(X_test.shape)
#print("Size of the test label")
#print(len(Y_test))

num_classes = 2
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(size, size,1), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 20
lrate = 0.01
decay = lrate/epochs

#load weights
model.load_weights("..//70weigths0.h5")

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(X_test.shape)
print(Y_test.shape)

score = model.evaluate(X_test, Y_test, verbose=0)
score2 = model.predict(X_test, verbose=0)
print(score2[:,1])
print("Accuracy: %.2f%%" % (score[1]*100))

print("Duree de l'execution : ")
print(time.time()-start)

a = 0
for k in range(0, 3):
    for i in range(len(liste_test)):
        try:
            img = cv2.imread("..//..//Data//VraiPhoto//" + str(liste_test[i]) + ".jpg", 1)
            if(score2[a][1] > 0.5):
                cv2.circle(img,(50,50), 50, (0,0,255), -1)
            cv2.imshow('Display window', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            a = a + 1
        except:
            print("Erreur : Pas d'image avec ces valeurs !")
