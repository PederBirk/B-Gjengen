import dataLoader as dl
import Network as NN
import pickle
import gzip


# prepare data
symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'X', '+', 'y']
pathToData = './extracted_images/'
nTrainingData = 10000
nTestData = 1000

# data = dl.loadData(symbols, nTrainingData, nTestData, pathToData)

# with open('kaggle-dataset.pkl', 'wb') as f:
# 	pickle.dump(data, f)

with gzip.open('./kaggle-dataset.pkl.gz', 'rb') as f:
	data = pickle.load(f)

# train network
net = NN.Network([2025,100, len(symbols)])
net.train(data['train'], 30, 100, 3.0, testData = data['test'])
