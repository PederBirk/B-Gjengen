import Network as nw

class Classifier:
	
	def __init__(self,symbols, network):
		self.symbols = symbols
		self.network = network

	def classify(self,char):
		activation= network.feedForward(char.image)
		char.symbol= getSymbol(activation) 
		
	def getSymbol(self,activation):
		maxactivation = activation[0]
		maxindex=0
		for i in range(1,len(activation)):
			if activation[i]>maxactivation:
				maxactivation=activation[i]
				maxindex=i
			
		return symbols[maxindex]
	
	def getProbSortedSymbols(self,char):
		activation=network.feedForward(char.image)
		zipped=zip(activation,range(len(activation)))
		zipped = sort(zipped)
		index = []
		for element in zipped:
			index.append(symbols[element[1]])
		return index
	