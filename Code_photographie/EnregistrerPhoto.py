# Fonction qui permet d'enregistrer une video et les images qui nous servirons a l'apprentissage

# Import des modules necessaires
import numpy as np
import cv2
import sys
import time

#Defini le nombre dephoto prise
nbrephoto = 827

# On fixe le nombre de pixel en hauteur et en largeur
pixelHauteur = 480
pixelLargeur = 640

# Permet de lire sur la webcam
cap = cv2.VideoCapture(1)

# On defini le type de la video pour enregistrer
fourcc = cv2.cv.CV_FOURCC('i', 'Y', 'U', 'V')

# On recupere la videoQ
out = cv2.VideoWriter('videoEnregistrer.avi',fourcc, 20.0, (640,480))

print(cap.isOpened())
# Boucle qui permet de lire la video, tant que la camera est allume
while(cap.isOpened()):
    # Defini le debut du timer qui calcul le temps de prise d'une image
    t0 = time.clock()

    # Lit la video
    ret, frame = cap.read()

    # On quitte le programme si on arrive pas a lire
    if ret:
        # On met l'image en gris
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # On affiche a l'ecran l'image
        out.write(frame)

        # on enregistre la video
        cv2.imshow('image',frame)

        # On redefini la taille pour enregistrer les images qui nous seront utiles
        frame = cv2.resize(frame, (pixelLargeur, pixelHauteur))

        # On enregistre l'image dans le bon dossier ATTENTION il faudra le changer
        cv2.imwrite("..//..//Data//2seancephoto//rien//"+str(nbrephoto)+".jpg", frame);

        # On rajoute une photo donc on incremente
        nbrephoto += 1

        # On attend l'appuie sur la touche "q" (quitter)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


    #qprint(time.clock()-t0)

    #Permet d'attendre le bontemps avant de continuer
    while(time.clock() - t0 < 0.5): #temps en seconde
        a=1
    print(time.clock() - t0)

# On relache tout
cap.release()
out.release()
cv2.destroyAllWindows()
print 'nombre de photo prise : ',nbrephoto + 1