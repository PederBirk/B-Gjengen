from equationParser import parseEquation
import imageProcessor as ip
import cv2
from ClassifierKeras import ClassifierKeras
from keras.models import model_from_json

img = ip.getImgMat('images/twoEq.jpg')
chars = ip.extractCharacters(img, (45,45))
lines = parseEquation(chars)

symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']

json_file = open('models/model_balanced.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model_balanced.h5")

classifier = ClassifierKeras(symbols, loaded_model)

for line in lines:
	for group in line:
		str = ""
		for c in group:
			classifier.classify(c)
			str += c.symbol
		print(str, end="\t")
	print("")

img = cv2.pyrDown(img)
cv2.imshow("test", img)
 
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
