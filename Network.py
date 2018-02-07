import numpy as np
import random

class Network():
	
	def __init__(self, nodesInLayer):
		self.numLayers = len(nodesInLayer)
		self.nodesInLayer = nodesInLayer
		self.biases = []
		self.weights = []
		for y in nodesInLayer[1:]:
			self.biases.append(np.random.randn(y,1)) #init biases with random numbers
			
		for x, y in zip(nodesInLayer[:-1], nodesInLayer[1:]):
			self.weights.append(np.random.randn(y,x)) #init weights with random numbers
		
	def feedForward(self, input): #Push data trough network, get prediction
		h = self.numLayers - 2
		a=[sigmoid(np.dot(self.weights,input)-self.biases[0])]
	
		for i in range(1,h+1):
			a.append(sigmoid(np.dot(self.weights[i],a[i-1])-self.biases[i]))
			return a[-1]
		
	#Main function for processing traning data and updating network using SGD
	def train(self, trainingData, epochs, batchSize, learningRate, testData = None):
		if testData : nTest = len(testData)
		n= len(trainingData)
		for j in range(epochs):
			random.shuffle(trainingData) 
			batches = []
			batches = [ trainingData[k:k + batchSize]
			                               for k in range(0, n, batchSize)] #Partiton training data into batches
			for batch in batches:
				self.updateBatch(batch, learningRate) #Run update batch for each batch
				
			print("epoch: ", j, " over")
			print("correct: ", self.evaluate(testData), "/", nTest)
		
	#Adjusts weights and biases in network for one batch
	def updateBatch(self, batch, learningRate):
		nablaB = [np.zeros(b.shape) for b in self.biases]
		nablaW = [np.zeros(w.shape) for w in self.weights]
		
		for input, expectedOutput in batch:
			deltaNablaB, deltaNablaW = self.backProp(input, expectedOutput) #Get gradient for each dataset in batch
			nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)] #Sum the gradients
			nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)] # ditto
			
		self.weights = [w-(learningRate/len(batch))*nw for w, nw in zip(self.weights, nablaW)] #Apply average gradient * learning rate
		self.biases = [b-(learningRate/len(batch))*nb for b, nb in zip(self.biases, nablaB)]
		
	#Calculates gradient of cost function
	def backProp(self, input, expectedOutput):
		nablaB = [np.zeros(b.shape) for b in self.biases]
		nablaW = [np.zeros(w.shape) for w in self.weights]
		
		activations = [input]
		activation = input
		zs = []
		
		for b, w in zip(self.biases, self.weights): #For each layer in network calculate weighted input(z) and corresponding activation
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
			
		delta = self.costDerivative(activations[-1], expectedOutput) * sigmoidPrime(zs[-1]) #Find derivative of cost function
		nablaB[-1] = delta
		nablaW[-1] = np.dot(delta, activations[-2].transpose())
		
		for layer in range(2, self.numLayers): #starting at 2nd to last layer, work backwards and find gradient for biases and weights
			z = zs[-layer]
			sp = sigmoidPrime(z)
			delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
			nablaB[-layer] = delta
			nablaW[-layer] = np.dot(delta, activations[-layer-1].transpose())
		
		return (nablaB, nablaW)
		
	#Runs test data and returns % of data correctly indentified
	def evaluate(self, testData):
		testResults = [(np.argmax(self.feedForward(x)),y) for (x,y) in testData]
		return sum(int(x==y) for (x,y) in testResults)
		
	#Returns partial derivatives
	def costDerivative(self, output, expectedOutput):
		return (output-expectedOutput)
	
	
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
	
def sigmoidPrime(z):
	return sigmoid(z)*(1-sigmoid(z))