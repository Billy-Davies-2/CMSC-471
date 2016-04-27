# Coded by:             William Davies
# Date:                 04/24/2016
# Class:                CMSC 471
# Description:
# This code will take an image file, and then convert it
# to black and white. Then it will convert the black and
# white image to a matrix representation.

# imports
from PIL import Image
from numpy import *
import sys
from sklearn.neighbors import NearestNeighbors

# convert an image to black and white.
def toBlackWhite(filename):
    image = Image.open(filename)
    image = image.convert('1')
    data = array(image)
    #data = [image.getpixel((x, y)) for x in range(image.width) for y in range(image.height)]
    bitmap = empty((data.shape[0],data.shape[1]),None)    #New array with same size as A

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j]==True:
                bitmap[i][j]=0
            else:
                bitmap[i][j]=1
    return bitmap

def main():
    filename = sys.argv[-1]
    bitmapMatrix = toBlackWhite(filename)
    print(bitmapMatrix)
main()