import Network as nw
import numpy as np
import cv2

class Classifier:
	
	def __init__(self, symbols, network):
		self.symbols = symbols
		self.network = network

	def classify(self,char):
		img = char.image
		img = np.transpose(img) #To match with mnist
		img = cv2.bitwise_not(img) #Same
		img = np.divide(img, 255.0)#Same

		print(np.array(np.reshape(img, (784, 1))))
		activation= self.network.feedForward(np.array(np.reshape(img, (784, 1))))
		char.symbol= self.getSymbol(activation) 
		
	def getSymbol(self,activation):
		maxactivation = activation[0]
		maxindex=0
		for i in range(1,len(activation)):
			if activation[i]>maxactivation:
				maxactivation=activation[i]
				maxindex=i
			
		return self.symbols[maxindex]
	
	def getProbSortedSymbols(self,char):
		activation=self.network.feedForward(char.image.flatten())
		zipped=zip(activation,range(len(activation)))
		zipped = sort(zipped)
		index = []
		for element in zipped:
			index.append(self.symbols[element[1]])
		return index
