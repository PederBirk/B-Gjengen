import Network
import mnist_loader

# updated for testing on the cluster; branch cluster 
net = Network.Network([784,342,171,57,10])
trainingData, validationData, testData = mnist_loader.load_data_wrapper()

net.train(trainingData, 30, 100, 3.0, testData = testData)
