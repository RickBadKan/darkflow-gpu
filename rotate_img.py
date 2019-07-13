from PIL import Image
import os
from bs4 import BeautifulSoup as Soup


def save_image(Imagem, nome):
	Imagem.save(nome)

def rename_images(path):
	os.getcwd()
	collection = path
	for i, filename in enumerate(os.listdir(collection)):
	    os.rename(collection + "/" + filename, collection + "/" + str(i) + ".jpg")


path_to_images = "./images"

rename_images(path_to_images)

os.getcwd()


for i in enumerate(os.listdir(path_to_images)):
	
	# Create an Image object from an Image
	colorImage = Image.open(path_to_images + "/" + str(i[0]) + ".jpg")

	# Rotate it by 90 degrees
	transposed90 = colorImage.transpose(Image.ROTATE_90)

	# Rotate it by 270 degrees
	transposed270 = colorImage.transpose(Image.ROTATE_270)

	save_image(transposed90, path_to_images + "/90_" + str(i[0]) + ".jpg")
	save_image(transposed270, path_to_images + "/270_" + str(i[0]) + ".jpg")