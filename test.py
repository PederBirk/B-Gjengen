import dataLoader as dl
import Network as NN
import pickle
import gzip
import imageProcessor as ip
from Classifier import Classifier


# prepare data
symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'X', '+', 'y']
pathToData = './data/'

# dl.pickleJPEGData(pathToData, symbols)

# data = dl.loadPickledData(pathToData, symbols)
# # train network
# net = NN.Network([2025, 100, 30, len(symbols)])
# net.train(data['train'], 30, 100, 3.0, testData = data['test'])

net = NN.load('net.json')
ip.webcam(Classifier(symbols, net))
