import dataLoader as dl
import Network as NN
import pickle
import gzip
import imageProcessor as ip
from ClassifierKeras import ClassifierKeras
from keras.models import model_from_json


# prepare data
symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
pathToData = './data/'

# dl.pickleJPEGData(pathToData, symbols)

# data = dl.loadPickledData(pathToData, symbols)
# rain network
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

ip.webcam(ClassifierKeras(symbols, loaded_model))
