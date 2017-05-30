import sys

print(sys.version)
import tensorflow
import keras
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
# from transform import dataset
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import KFold
import time
import theano
import numpy.ma as ma
import matplotlib.cm as cm
from keras import backend as K

#def convout1_f(X):
    # The [0] is to disable the training phase flag
#    return _convout1_f([0] + [X])

start = time.time()

from mpl_toolkits.axes_grid1 import make_axes_locatable


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)



def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    #nimgs = 32
    imshape = imgs.shape[1:]
    #imshape = (70, 70)
    #print(nimgs, imshape)

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def inverse_matrix(C1):
    image_conv = np.empty((32, 70, 70))
    for pi in range(70):
        for pj in range(70):
            for pk in range(32):
                image_conv[pk, pi, pj] = C1[pi, pj, pk]
    return (image_conv)

def kfolds(k, N, seed=None):
    random.seed(seed)
    out = [list() for _ in range(k)]
    for n in range(N):
        out[random.randrange(k)].append(n)
    return (out)


# https://github.com/fchollet/keras/issues/4465 for VGG16 model

nombrePhoto0 = 180
nombrePhoto1 = 100
nombrePhoto2 = 100

nombrePhoto = [nombrePhoto0, nombrePhoto1, nombrePhoto2]

# resize image
size = 70

resultat = []
resultatParam = []
resultatFauxPositif = []
resultatFauxNegatif = []
categorie = ["rien", "cylindrejaune", "cylindrebleu"]

n_split = 5

Kf = kfolds(seed=486684, k=n_split, N=max(nombrePhoto))

scores = []

for n in range(0, 1):
    Y_train = []
    X_train = []
    Y_test = []
    X_test = []
    print(n)

    list1 = [i for i in range(0, n_split)]
    list1.remove(n)
    train = []
    for j in list1:
        train = sum(Kf[j:j + 1], train)
    test = Kf[n]

    for k in range(0, 3):
        for i in train:
            try:
                Img = Image.open("Data//2seancephoto//" + categorie[k] + "//" + str(i) + ".jpg")
                # Img = Image.open("//home//tanguy//Documents//Cassiopee//NeuronalNetwork//Data//2seancephoto//"+categorie[k]+"//" + str(i) + ".jpg")
                # Img = Img.convert('LA')
                Img = Img.resize((size, size), PIL.Image.ANTIALIAS)
                # Img = list(Img.getdata())
                Img = img_to_array(Img)
                # Img = np.array(Img)
                # Img = Img[:, 0]
                # Img = Img.reshape(size, size, 1)
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
                Img = Image.open("Data//2seancephoto//" + categorie[k] + "//" + str(i) + ".jpg")
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
            except:
                blabla = 1

    print("On a creer X")
    print(Y_test)
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

    # Let's build The CNN!!!!!
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(size, size, 3), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    convout1 = Dropout(0.2)
    model.add(convout1)
    # model.add(Dropout(0.2))
    convout2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 10
    lrate = 0.01
    decay = lrate / epochs
    # keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # load weights
    # model.load_weights("..//weigths0.h5")

    sgd = SGD(lr=lrate, momentum=0.5, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print("let's take a look at the architecture of the Network")
    print(model.summary())

    # let's fit our model

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32)
    # model.save_weights("..//70weigths_color"+str(n)+".h5")
    # Final evaluation of the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (score[1] * 100))
    scores.append(score[1] * 100)
    print(time.time() - start)

    # K.learning_phase() is a flag that indicates if the network is in training or
    # predict phase. It allow layer (e.g. Dropout) to only be applied during training
    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, [convout1.output])
    C1 = _convout1_f([0] + [X_train])
    C1 = np.squeeze(C1)
    print("C1 shape : ", C1.shape)

    plt.figure(figsize=(15, 15))
    plt.suptitle('convout1')

    photonum = 250
    C1_1 = C1[photonum, :, :, :]
    image_conv_1 = inverse_matrix(C1_1)

    nice_imshow(plt.gca(), make_mosaic(image_conv_1, 6, 6), cmap=cm.binary)
    plt.show()



    inputs = [K.learning_phase()] + model.inputs
    _convout2_f = K.function(inputs, [convout2.output])
    C1 = _convout2_f([0] + [X_train])
    C1 = np.squeeze(C1)
    print("C1 shape : ", C1.shape)

    plt.figure(figsize=(15, 15))
    plt.suptitle('convout2')

    photonum = 250
    C1_1 = C1[photonum, :, :, :]
    image_conv_2 = inverse_matrix(C1_1)

    nice_imshow(plt.gca(), make_mosaic(image_conv_2, 6, 6), cmap=cm.binary)
    plt.show()

print(scores)
