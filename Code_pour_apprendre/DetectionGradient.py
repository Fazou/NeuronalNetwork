
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


img0 = cv2.imread("..//Data//2seancephoto//rien//1.jpg", 1)
a = img0.shape
tailleImage = a[0] * a[1] * a[2]


img0 = cv2.cvtColor( img0, cv2.COLOR_RGB2GRAY )

cv2.imshow('Display window', img0);

while True:
    a = cv2.waitKey(1) # si l'utilisateur appuye sur une touche, on stock le resultat dans a
    if(a==1048689):
        cv2.destroyAllWindows()
        break

print("impression de la suite:")
Ex= np.gradient(img0)
print(len(Ex[0][0]))
print(img0)
A = np.sqrt(Ex[0]**2 + Ex[1]**2)

cv2.imshow('Display window', Ex[0]);
while True:
    a = cv2.waitKey(1) # si l'utilisateur appuye sur une touche, on stock le resultat dans a
    if(a==1048689):
        cv2.destroyAllWindows()
        break

time.sleep(15)
while(True):
    a=4


img = img0.reshape(1, tailleImage)

img = np.concatenate((img,np.array([[1]])),axis=1)
#print(img)
target = [1]

#gradient = Gradient()
#print(gradient)


print("On a vu la photo maintenant on va tout importer")

for i in range(1,nombrePhoto1):
    image = cv2.imread("..//Data//Label1//"+ str(i) + ".jpg", 1)
    img0 = image.reshape(1,tailleImage)
    img0 = np.concatenate((img0, np.array([[1]])), axis=1)
    img = np.concatenate((img, img0), axis=0)
    target.append(1)
for i in range(0,nombrePhoto0):
    image = cv2.imread("..//Data//Label0//"+ str(i) + ".jpg", 1)
    img0 = image.reshape(1,tailleImage)
    img0 = np.concatenate((img0, np.array([[0]])), axis=1)
    img = np.concatenate((img, img0), axis=0)
    target.append(0)

target = np.array(target)
print(img)

kf = KFold(n_splits=5)
for train, test in kf.split(img):
    print("%s %s" % (train, test))


# Create a classifier: a support vector classifier
#classifier = svm.SVC(gamma=0.001,kernel='rbf')
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# We learn the digits on the first half of the digits
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