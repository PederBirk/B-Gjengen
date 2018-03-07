import Network as nw
import imageProcessor as ip
import Classifier as cl
import numpy as np
import cv2
import mnist_loader

imgMat = ip.getImgMat('ex.png')
chars = ip.extractCharacters(imgMat, (28,28))

#net = nw.Network([784,100,10])
trainingData, validationData, testData = mnist_loader.load_data_wrapper()
#net.train(trainingData, 3 ,10, 3.0, testData = testData)
net = nw.load('mnist.json')

a = testData[0]
	
print(a)
#print(imgMat)

clas = cl.Classifier(['0','1','2','3','4','5','6','7','8','9'], net)
for c in chars:
	clas.classify(c)
	print(c.symbol)
	cv2.putText(imgMat, c.symbol, (c.xPos,c.yPos), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)

cv2.imshow("test", imgMat)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()