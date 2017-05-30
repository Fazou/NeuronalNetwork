
#from transform import dataset
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import KFold
import time
import cv2
import os




size = 70
x,y=400,500
h,w=480 - 1  + y,640-1 + x

file_names = os.listdir("..//Data//table_final_tournoi//cylindrebleu")
print(file_names)

for k in range(len(file_names)):
    image = cv2.imread("..//Data//table_final_tournoi//cylindrebleu//"+file_names[k], 1)
    cv2.imwrite("..//Data//table_final_tournoi//cylindrebleu//"+str(k)+".jpg",image)


#file_names = sorted(file_names, key=lambda
#    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))


#pyplot.imshow("..//Data//table_final_tournoi//pastraiter//" + file_names[0])
#imagemat = cv2.imread("..//Data//table_final_tournoi//pastraiter//" + file_names[0], 1)

#print(file_names[4])
#image = cv2.imread("..//Data//table_final_tournoi//pastraiter//IMG_20170525_145904.jpg", 1)
#cv2.imshow('Display window', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(image)
#height, width, channels = image.shape
#print (height, width, channels)
#a = 25
for i in range(0,0):
    for j in range(0,0):
        x, y = 1810+ i*20,1690 + j*20
        h, w = 48*6 - 1 + y, 64*6 - 1 + x
        print(x,y,h,w)
        imagetest = image[y:h, x:w]
        cv2.imshow('Display window', imagetest)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("..//Data//table_final_tournoi//cylindrejaune//"+str(a)+".jpg",imagetest)
        a = a + 1