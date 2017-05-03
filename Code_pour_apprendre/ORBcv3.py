import cv2

x,y=0+75,25
w,h=680-125-1,480-25-1
#613
#273
#1013

#29
#23 244
#749


#287 373

image = cv2.imread("..//Data//2seancephoto//cylindrebleu//749.jpg", 1)

image = image[y:h, x:w]  # Crop from x, y, w, h -> 100, 200, 300, 400

cv2.imshow('Display window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
