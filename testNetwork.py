import Network
import mnist_loader

net = Network.Network([784,100,10])
trainingData, validationData, testData = mnist_loader.load_data_wrapper()
net.train(trainingData, 30 ,1, 3.0, testData = testData)