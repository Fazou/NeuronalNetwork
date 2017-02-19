print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

import time


N = 400

# The digits dataset
digits = datasets.load_digits()

#print(len(digits.images))
#print(len(digits[0]))

#print(digits.images[0])

img = cv2.imread("//home//tanguy//Documents//Cassiopee//Data//0.jpg", 1)

cv2.imshow('Display window', img);

while True:
    a = cv2.waitKey(1) # si l'utilisateur appuye sur une touche, on stock le resultat dans a
    if(a==1048689):
        break
print("fini")
print(img.shape)
#print(img)


# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
print("n_sample : ")
#print (n_samples)
#print(digits.images)
print(digits.images.shape)

data = digits.images.reshape((n_samples, -1))
print(data.shape)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001,kernel='rbf')
# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples - N], digits.target[:n_samples - N])
# Now predict the value of the digit on the second half:
expected = digits.target[N:]
predicted = classifier.predict(data[N:])
print("Classification report for classifier yolo %syolo:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
images_and_predictions = list(zip(digits.images[N:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()
