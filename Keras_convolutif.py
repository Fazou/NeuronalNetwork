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

start = time.time()

def kfolds(k, N, seed=None):
    random.seed(seed)
    out = [list() for _ in range(k)]
    for n in range(N):
        out[random.randrange(k)].append(n)
    return (out)


#https://github.com/fchollet/keras/issues/4465 for VGG16 model

nombrePhoto0 = 1700
nombrePhoto1 = 1000
nombrePhoto2 = 1000

nombrePhoto = [nombrePhoto0,nombrePhoto1,nombrePhoto2]

#resize image
size = 70

resultat=[]
resultatParam = []
resultatFauxPositif=[]
resultatFauxNegatif=[]
categorie = ["rien", "cylindrejaune", "cylindrebleu"]

n_split = 5


K = kfolds(seed=486684,k=n_split,N=max(nombrePhoto))


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
        for i in train:
            try:
                Img = Image.open("Data/2seancephoto/"+categorie[k]+"/" + str(i) + ".jpg")
                #Img = Image.open("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//2seancephoto//"+categorie[k]+"//" + str(i) + ".jpg")
                Img = Img.convert('LA')
                Img = Img.resize((size, size), PIL.Image.ANTIALIAS)
                Img = list(Img.getdata())
                Img = np.array(Img)
                Img = Img[:, 0]
                Img = Img.reshape(size, size, 1)
                X_train.append(Img)

                if (k == 0):
                    Y_train.append(0)
                if (k == 1):
                    Y_train.append(1)
                if (k == 2):
                    Y_train.append(1)
            except:
                blabla = 1


        for i in test:
            try:
                Img = Image.open("Data/2seancephoto/"+categorie[k]+"/" + str(i) + ".jpg")
                #Img = Image.open("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//2seancephoto//"+categorie[k]+"//" + str(i) + ".jpg")
                Img = Img.convert('LA') #Pour noir et blanc
                Img = Img.resize((size, size), PIL.Image.ANTIALIAS)
                Img = list(Img.getdata()) #Pour noir et blanc sinon mettre : Img = img_to_array(Img)
                Img = np.array(Img)
                Img = Img[:, 0] #Que noir et blanc !
                Img = Img.reshape(size, size, 1) #Que noir et blanc encore !
                X_test.append(Img)

                if (k == 0):
                    Y_test.append(0)
                if (k == 1):
                    Y_test.append(1)
                if (k == 2):
                    Y_test.append(1)
            except:
                blabla = 1

    print("On a creer X")

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0



    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)


    print("Size of training dataset")
    print(X_train.shape)
    print("Size of the training label")
    print(len(Y_train))
    print("Size of test dataset")
    print(X_test.shape)
    print("Size of the test label")
    print(len(Y_test))





    #Let's build The CNN!!!!!
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
    #model.load_weights("..//weigths0.h5")

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print("let's take a look at the architecture of the Network")
    print(model.summary())

    #let's fit our model

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32)
    model.save_weights("..//70weigths"+str(n)+".h5")
    # Final evaluation of the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (score[1]*100))
    scores.append(score[1]*100)
    print(time.time()-start)

print(scores)