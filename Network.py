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
			
	def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))
		
	def feedForward(self, input): #Push data trough network, get prediction
		return None
		
	#Main function for processing traning data and updating network using SGD
	def train(self, trainigData, epochs, batchSize, learningRate, testData = None):
		n= len(trainigData)
		for j in xrange(epochs):
			random.shuffle(trainigData) 
			batches = []
			for k in xrange(0, n, batchSize):
				batches.append[trainigData[k:k+batchSize]] #Partiton training data into batches
			for batch in batches:
				self.updateBatch(batch, learningRate) #Run update batch for each batch
		
	#Adjusts weights and biases in network for one batch
	def updateBatch(self, batch, learningRate):
		nablaB = [np.zeros(b.shape) for b in self.biases]
		nablaW = [np.zeros(w.shape) for w in self.weights]
		
		for input, expectedOutput in batch:
			deltaNablaB, deltaNablaW = backProp(input, expectedOutput) #Get gradient for each dataset in batch
			nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)] #Sum the gradients
			nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)] # ditto
			
		for layer in xrange(len(self.weights)):
			avgGradient = nablaW[layer]/len(batch) 
			self.weights[layer] = self.weights[learningRate] - avgGradient*learningRate #Apply average gradient * learning rate
		
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
		nablaW[-1] = np.dot(delta, activations[-2].traspose())
		
		for layer in xrange(2, self.numLayers): #starting at 2nd to last layer, work backwards and find gradient for biases and weights
			z = zs[-layer]
			sp = sigmoidPrime(z)
			delta = np.dot(self.weights[-layer+1].traspose(), delta) * sp
			nablaB[-layer] = delta
			nablaW[-layer] = np.dot(delta, activations[-layer-1].traspose())
		
		return (nablaB, nablaW)
		
	#Runs test data and returns % of data correctly indentified
	def evaluate(self, testData):
		return None
		
	#Returns partial derivatives
	def costDerivative(self, output, expectedOutput):
		return (output-expectedOutput)
		
	def sigmoidPrime(z):
		return sigmoid(z)*(1-sigmoid(z))
	
	
		