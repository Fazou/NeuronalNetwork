from numpy import *
import numpy as np
import cv2
import time

x,y,z = mgrid[-2:0:1., -5:1:1., -2:1:1.]
#x = mgrid[-5:0:1.]

V = 2*x**2 + 3*y**2 - 4*z # just a random function for the potential
#print(V)
#print(x)
Z = np.array([[1,2,3,4,5,5,5,5],[16,2,3,4,5,5,5,5],[1,5,6,6,5,5,5,5]])
print(Z)
Ex,Ey,Ez = np.gradient(V)
print(np.gradient(Z))
#print(Ez)
#print(y)
#print(z)


img = cv2.imread("..//Data//2seancephoto//rien//1.jpg", 1)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

time.sleep(5)