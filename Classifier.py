import Network as nw
import numpy as np
import cv2

class Classifier:
	
	def __init__(self, symbols, network):
		self.symbols = symbols
		self.network = network

	def classify(self,char):
		activation= self.network.feedForward(char.image.reshape((2025,1))/255)
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
		activation= self.network.feedForward(char.image)
		zipped=zip(activation,range(len(activation)))
		zipped.sort()
		index = []
		for element in zipped:
			index.append(self.symbols[element[1]])
		return index
