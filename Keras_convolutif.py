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


#https://github.com/fchollet/keras/issues/4465 for VGG16 model


#resize image
size = 50
nbre_image = 170
nbre_train = 120
nbre_test = 50

Img_jaune=[]
for i in range(nbre_image):
    Img = Image.open("../Data/2seancephoto/cylindrejaune/"+str(i)+".jpg")
    Img = Img.resize((size,size ), PIL.Image.ANTIALIAS)
    Img = img_to_array(Img)
    Img = np.array(Img)
    Img_jaune.append(Img)
a = Img_jaune[0]
Img_jaune = np.array(Img_jaune)
#plt.imshow(a)
#plt.show()

Img_rien=[]
for i in range(nbre_image):
    Img_1 = Image.open('Data/2seancephoto/rien/'+str(i)+'.jpg')
    Img_1 = Img_1.resize((size,size ), PIL.Image.ANTIALIAS)
    Img_1 = img_to_array(Img_1)
    Img_1 = np.array(Img_1)
    Img_rien.append(Img_1)
b = Img_rien[0]
Img_rien = np.array(Img_rien)
#plt.imshow(b)
#plt.show()

data = np.concatenate((Img_jaune,Img_rien), axis=0)
Y_train=[]
Y_test = []
X_train_1 = data[:nbre_train,]
for i in range(nbre_train):
    Y_train.append(1)
X_test_1 = data[nbre_train:nbre_test+nbre_train+1,]
for i in range(nbre_test):
    Y_test.append(1)
X_train_2 = data[nbre_test+nbre_train+1:nbre_test+2*nbre_train+1,]
for i in range(nbre_train):
    Y_train.append(0)
X_test_2 = data[nbre_test+2*nbre_train+1:,]
for i in range(nbre_test):
    Y_test.append(0)
X_train = np.concatenate((X_train_1,X_train_2), axis=0)
X_test = np.concatenate((X_test_1,X_test_2), axis=0)

# normalize inputs from 0-255 to 0.0-1.0
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

""""
#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(200,200,3),name = 'image_input')

#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)

#Create your own model
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
lrate = 0.01
epochs = 1
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
my_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
my_model.summary()
my_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32)
my_model.save_weights('../../weigths_150.h5')"""





#Let's build The CNN!!!!!
num_classes = 2
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(size, size, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
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
#model.load_weights("../weigths_20.h5")

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print("let's take a look at the architecture of the Network")
print(model.summary())

#let's fit our model

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=32)
#model.save_weights('weigths_20.h5')
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




img_rows = 480
img_cols = 640
img_depth = 3

"""img_rien = dataset('Data/2seancephoto/rien/').data[:20]
img_jaune = []
for i in range(20):
    path = 'Data/2seancephoto/cylindrejaune/'+str(i)+'.jpg'
    img = load_img(path)
    img_jaune.append(img_to_array(img))
img_jaune = np.array(img_jaune)"""

"""img = load_img('Data/2seancephoto/cylindrejaune/0.jpg')  # this is a PIL image
#x = img_to_array(img)  # this is a Numpy array with shape (480, 640, 3)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 480, 640, 3)"""

#1ere couche
"""model = Sequential()
model.add(Conv2D(32, (2, 50), input_shape=(3, 480, 640))) #32 reduction colonne et (3,3) reduction profondeur de 3-1 et ligne de 3-1
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #(2,2) reduction profondeur 2-1 et division ligne par 2
print(model.output_shape)"""

"""#2eme couche
model.add(Conv2D(100, (2, 3)))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

#3eme couche
model.add(Conv2D(64, (2, 3)))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))"""


"""model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Activation('sigmoid'))"""

"""model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])"""


