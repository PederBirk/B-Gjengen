import cv2
import numpy as np
import gzip
import pickle
from math import floor
from os import listdir

# reads raw image data from inPath into a two lists (train and test) of tuples consisting of
# input data and correct output; for training data the correct output is vectorized, but for test data
# it is the index of the symbol for compatibility with feedforward's output
def pickleJPEGData(inPath, symbols): # TODO validation data?
	data = { 'train': [], 'test': [] }
	for symbol in symbols:
		path = inPath + symbol + '/'
		files = [ readImage(inPath + symbol + '/' + f) for f in listdir(path) ]
		nTrain = floor(len(files) * 0.9)
		data['train'] = [(trainInput, vectorize(symbol, symbols)) for trainInput in files[:nTrain]]
		data['test'] = [(testInput, vectorize(symbol, symbols)) for testInput in files[nTrain:]]
		with open(inPath + symbol + '.pkl', 'wb') as f:
			pickle.dump(data, f)

#Create unit vector representation of symbol. Indexing is determined by 'symbols' list
def vectorize(symbol, symbols):
	output = np.zeros(len(symbols))
	output[symbols.index(symbol)] = 1
	return output

#Read a single image into a binary column vector
def readImage(path):
	img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	# tresh, binImg = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	return img

def loadPickledData(path, symbols):
	data = { 'training-input': [], 'training-output': [], 'test-input': [], 'test-output': [] }
	print("loading symbols...")
	for symbol in symbols:
		with open(path + symbol + '.pkl', 'rb') as f:
				symbolData = pickle.load(f)
		data['training-input'].extend([inp for inp, outp in symbolData['train']])
		data['training-output'].extend([outp for inp, outp in symbolData['train']])
		data['test-input'].extend([inp for inp, outp in symbolData['test']])
		data['test-output'].extend([outp for inp, outp in symbolData['test']])
	data['training-input'] = np.array(data['training-input'])
	data['training-output'] = np.array(data['training-output'])
	data['test-input'] = np.array(data['test-input'])
	data['test-output'] = np.array(data['test-output'])
	return data
