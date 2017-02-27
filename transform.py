import numpy as np
from PIL import Image
import os

"""img = Image.open( "1.jpg" )
try:
    data = np.asarray( img, dtype='uint8' )
except SystemError:
    data = np.asarray( img.getdata(), dtype='uint8' )
print(data)"""	

class dataset:
	""" transform the image dataset into an array dataset """
	img = []
	data = []
	target = []
	# We build our constructor
	def __init__(self, folderpath):
		for fichier in os.listdir(folderpath):
			if fichier != 'Thumbs.db':
				dataset.img.append(Image.open(str(folderpath) + '/' + fichier))
				dataset.target.append(int(fichier.replace(".jpg", "")))
		for i in dataset.img:
			try:
				dataset.data.append(np.asarray( i, dtype='uint8' ))
			except SystemError:
				dataset.data.append(np.asarray( i.getdata(), dtype='uint8' ))
		
	 
				
image = dataset('C:/Users/lassana/Anaconda2/Machine_Learning/images')
print(image.data) 
print(image.target)
