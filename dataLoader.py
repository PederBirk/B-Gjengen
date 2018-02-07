import cv2
import numpy as np
from os import listdir

#Read data into list of tuples (imagePixels, expectedOutput)
#imagePixels is a binary column vector, expectedOutput is a unit column vector
#Retruns two lists: trainingData and testData
def loadData(symbols, nTrainingData, nTestData, pathToData):

    trainingData = []
    testData = []
    
    for symbol in symbols:
        trainingCounter = 0
        testCounter = 0
        expectedOutput = vectorize(symbol,symbols)
        path = pathToData + symbol
        for file in listdir(path):
            if trainingCounter < nTrainingData/len(symbols):
                trainingData.append((readImage(pathToData + symbol + '/'  + file), expectedOutput))
                trainingCounter += 1
            elif testCounter < nTestData/len(symbols):
                testData.append((readImage(pathToData + symbol + '/'  + file), expectedOutput))
                testCounter += 1
            else:
                break
    return (trainingData, testData)

#Create unit vector representation of symbol. Indexing is determined by 'symbols' list
def vectorize(symbol, symbols):
    output = np.zeros((len(symbols),1))
    output[symbols.index(symbol)] = 1
    return output

#Read a single image into a binary column vector
def readImage(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    tresh,binImg = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binImg.flatten()