

import cv2
import numpy as np
import time

img = cv2.imread("..//Data//2seancephoto//cylindrejaune//350.jpg",1)

cv2.imwrite('houghlines3.jpg',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Display window', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
edges = cv2.Canny(gray,50,70,apertureSize = 3,L2gradient=True)
cv2.imwrite('houghlines3.jpg',edges)

cv2.imshow('Display window', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLines(edges,100,np.pi/180,90)

#print(cv2.HoughLines(edges,1,np.pi/180,200))
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

"""minLineLength = 1000
maxLineGap = 0.1
lines = cv2.HoughLinesP(edges,1,np.pi/180,90,minLineLength,maxLineGap)
#lines = cv2.HoughLinesP(edges,1,np.pi/180,90,minLineLength)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)"""

print(lines)


cv2.imwrite('houghlines3.jpg',img)
cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()