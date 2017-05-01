import cv2
import numpy as np
from sklearn import mixture

#2.machin !


print(cv2.__version__)
# Load the images

X = []
for i in range(0,2):
    print(i)
    # Convert them to grayscale
    imgg = cv2.imread("..//Data//2seancephoto//cylindrejaune//"+str(i)+".jpg", 1)
    imgg = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)

    # SURF extraction
    orb = cv2.ORB()

    kp = orb.detect(imgg,None)
    kp, des = orb.compute(imgg, kp)
    img2 = cv2.drawKeypoints(imgg,kp,color=(0,255,0), flags=0)

    cv2.imshow('tm', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector = cv2.FeatureDetector_create("ORB") #SURF
    descriptor = cv2.DescriptorExtractor_create("ORB") #BRIEF

    kp_scene = detector.detect(imgg)

    k_scene, d_scene = descriptor.compute(imgg, kp_scene)
    print ('#keypoints in image1: %d' % len(d_scene))
    for j in range(0,len(d_scene)):
        X.append(d_scene[j])

gmm = mixture.GaussianMixture(n_components=10, covariance_type='full').fit(X)

print(gmm.predict_proba(X[0].reshape(1, -1)))
print(sum(gmm.predict_proba(X)))


for i in range(0,1):
    a = 4
