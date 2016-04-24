# Coded by: 	William Davies
# Date:			04/24/2016
# Class:		CMSC 471
# Description:
# This code will take an image file, and then convert it
# to black and white. Then it will convert the black and
# white image to a matrix representation.

# imports
from PIL import Image 

# convert an image to black and white.
def toBlackWhite(filename):
	imageFile = Image.open(filename)
	imageFile.convert('1')
	imageFile.save(filename+'_convert.png')
	toBitmap(filename+'_convert.png')

# convert black and white image to bitmap
def toBitmap(filename):
	print("attempting to convert: ", filename)