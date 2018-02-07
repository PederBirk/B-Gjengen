import cv2
import numpy as np
from os import listdir
import os.path


symbols = ['0','1','2','3','4','5','6','7','8','9']
pathToData = '../extracted_images/'

data = []

files = listdir(pathToData)

for symbol in symbols:
    expectedOutput = vectorize(symbol,symbols)
    path = pathToData + symbol
    for file in listdir(path):
        data.append(zip(readImage(file),expectedOutput))


def vectorize(symbol, symbols):
    output = np.zeros((len(symbols),1))
    output[symbols.index(symbol)] = 1
    return output

def readImage(path):
    img = cv2.pyrDown(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)
    img.reshape(img.size,1)
    return img
 