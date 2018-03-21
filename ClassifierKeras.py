import Network as nw
import numpy as np

class ClassifierKeras:
	
	def __init__(self, symbols, network):
		self.symbols = symbols
		self.network = network

	def classify(self,char):
		activation= self.network.predict(np.reshape(char.image/255,(1,45,45,1)))
		activation = np.reshape(activation,(len(self.symbols)))
		char.symbol= self.getSymbol(activation)
		
	def getSymbol(self,activation):
		maxactivation = activation[0]
		maxindex=0
		for i in range(1,len(activation)):
			if activation[i]>maxactivation:
				maxactivation=activation[i]
				maxindex=i
		if maxactivation < 0.6: return "?"
		return self.symbols[maxindex]
	
	def getProbSortedSymbols(self,char):
		activation= self.network.predict(char.image/255)
		zipped=zip(activation,range(len(activation)))
		zipped.sort()
		index = []
		for element in zipped:
			index.append(self.symbols[element[1]])
		return index
