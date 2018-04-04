import dataLoader as dl
import Network as NN
import pickle
import gzip
import imageProcessor as ip
from ClassifierKeras import ClassifierKeras
from keras.models import model_from_json
import cv2

symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
pathToData = './data/'

# dl.pickleJPEGData(pathToData, symbols)

# data = dl.loadPickledData(pathToData, symbols)
# rain network
json_file = open('model_shuffled.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_shuffled.h5")
classifier = ClassifierKeras(symbols,loaded_model)

imgPath = 'C:\\Users\\t_tor\\Unsynced\\extracted_images\\2\\exp32472 .jpg'
img = ip.getImgMat(imgPath)
cv2.imshow("test", img)
if cv2.waitKey(0) & 0xff == 27:
	 cv2.destroyAllWindows()
chars = ip.extractCharacters(img)
classifier.classify(chars[0])
print(chars[0].symbol)
imgRect = img.copy()
ip.drawCharacterBounds(imgRect,chars)
ip.drawClassifiedCharacters(imgRect, classifier, chars)
cv2.imshow("Classifications", imgRect)
if cv2.waitKey(0) & 0xff == 27:
	 cv2.destroyAllWindows()
chars = ip.extractCharacters(img)
for char in chars:
	char.show()
	key = cv2.waitKey(20)
	if key == 27: # ESC
		break
cv2.destroyAllWindows()