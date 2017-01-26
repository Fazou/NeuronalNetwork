# Fonction qui permet d'enregistrer les images qui nous servirons a l'apprentissage

# Import des modules necessaires
import numpy as np
import cv2
import sys
import time

# Defini le nombre de photo prise
nbrephoto = 0

# On fixe le nombre de pixel en hauteur et en largeur
pixelHauteur = 50
pixelLargeur = 75

# Valeur pour rogner l'image
premierPixelHauteur = 0
dernierPixelHauteur = 75
premierPixelLargeur = 0
dernierPixelLargeur = 50

# Boolean qui defini si la personne a appuyer sur q
stop = True


# Un tour de boucle prend une photo, on arrete quand l utilisateur a appuyer sur "q"
while(stop):

    # Permet de lire sur la webcam
    cap = cv2.VideoCapture(0)

    # On defini le type de la video pour enregistrer
    fourcc = cv2.cv.CV_FOURCC('i', 'Y', 'U', 'V')

    # On recupere la video
    #out = cv2.VideoWriter('videoEnregistrer.avi', fourcc, 20.0, (640, 480))

    # Lit la video
    ret, frame = cap.read()

    # On quitte le programme si on arrive pas a lire
    if ret:

        # On met l'image en gris
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # On affiche l'image et on attend que l'utilisateur indique qu'ils faut passer a la prise suivante
        while True:
            ret, frame = cap.read()

            #On passe en noir et blanc
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # On redefini la taille pour enregistrer les images qui nous seront utiles
            #frame = cv2.resize(frame, (pixelLargeur, pixelHauteur))

            # Permet de rogner l'image
            #frame = frame[premierPixelLargeur:dernierPixelLargeur, premierPixelHauteur:dernierPixelHauteur]

            cv2.imshow('image', frame) #On affiche l image
            a = cv2.waitKey(1) # si l'utilisateur appuye sur une touche, on stock le resultat dans a
            if(a==115): # 115 correspond a la lettre q
                break
            if(a==113): # 113 correspond a la lettre s
                stop = False
                break

        # On redefini la taille pour enregistrer les images qui nous seront utiles
        frame = cv2.resize(frame, (pixelLargeur, pixelHauteur))

        #Permet de rogner l'image
        frame = frame[premierPixelLargeur:dernierPixelLargeur, premierPixelHauteur:dernierPixelHauteur]

        # On enregistre l'image dans le bon dossier ATTENTION il faudra le changer
        cv2.imwrite("C:\Users\Tanguy\Documents\TSP\OpenCV_python\Webcam-Face-Detect-master\Gray_Image"+str(nbrephoto)+".jpg", frame);

        # On rajoute une photo donc on incremente
        nbrephoto += 1
        print 'nombre de photo prise : ', nbrephoto

    else:
        break

    # On relache tout
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
print 'nombre de photo final prise : ',nbrephoto