import dataLoader as dl
import Network as NN
import pickle
import gzip


# prepare data
symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'X', '+', 'y']
pathToData = './extracted_images/'
nTrainingData = 10000
nTestData = 1000

# dl.pickleJPEGData(pathToData, symbols)

data = dl.loadPickledData('./extracted_images/', symbols)

# train network
net = NN.Network([2025, 100, 30, len(symbols)])
net.symbols = symbols
net.train(data['train'], 30, 100, 3.0, testData = data['test'])
