# import the necessary packages
import numpy as np
import argparse
import cv2

# load the image
image = cv2.imread("//home//tanguy//Documents//Cassiopee//Data//1seancephoto//cylindrejaune//20.jpg")
cv2.circle(image, (400,300), 10,(0,84,96),-1)
# define the list of boundaries
boundaries = [
	([0, 50, 50], [20, 150, 150]),
	#([86, 31, 4], [220, 88, 50]),
	#([25, 146, 190], [62, 174, 250]),
	#([103, 86, 65], [145, 133, 128])
]

# loop over the boundaries
if __name__ == '__main__':
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        print(mask)
        output = cv2.bitwise_and(image, image, mask=mask)

        # show the images
        cv2.imshow("images", np.hstack([image, output]))
        cv2.waitKey(0)
    #IDEE : on regarde une colone de mask, et on met le nombre de 1 dans cette colone. En gros on somme les colones
    # C est ce vecteurs qu on va itiliser pour la pprentissage, peut etre pas prendre troute
    # les colones, mais des paquets de 5 par exemples.
    # Ainsi on a un vecteur
    # Attention cette methode ne permet que de faire jaune/pasjaune :/ Mais ca peut etre stylee
