#!/usr/bin/python

import cv2, sys
import operator

files = sys.argv[1:]
all_ratio = {}
for i in range(0,1000):

    try:
        orig = cv2.imread("..//Data//2seancephoto//cylindrejaune//" + str(i) + ".jpg", 1)


        sobel_dx = cv2.Sobel(orig, cv2.CV_64F, 1, 0, ksize=5)
        sobel_dy = cv2.Sobel(orig, cv2.CV_64F, 0, 1, ksize=5)
        magnitude_image = cv2.magnitude(sobel_dx,sobel_dy,sobel_dx);
        mag, ang = cv2.cartToPolar(sobel_dx, sobel_dy, magnitude_image)

        ratio = cv2.sumElems(mag[0])
        all_ratio[i] = ratio[0]

    except:
        print("coucou")

sorted_ratio = sorted(all_ratio.items(), key=operator.itemgetter(1))
index = 1
print(" Rang | Fichier      | Valeur calculee")
print("------|--------------|----------------")
for (filename, ratio) in reversed(sorted_ratio):
    print(" %04d | %s | %d" % (index, filename, ratio))
    index += 1