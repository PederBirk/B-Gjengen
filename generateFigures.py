import cv2
import imageProcessor as ip
from Character import Character
import characterProcessor as cp
from ClassifierKeras import ClassifierKeras
from keras.models import model_from_json
from equationParser import parseEquation
from equationParser import drawSolvedEquations

path = 'images/twoEq.jpg'
symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
	
json_file = open('models/model_balanced.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model_balanced.h5")
classifier = ClassifierKeras(symbols, loaded_model)


img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
cv2.imwrite("figures/original.jpg", img)

binImg = ip.getImgMat(path)
cv2.imwrite("figures/threshed.jpg", binImg)

image, contours, hier = cv2.findContours(binImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
imgHeight, imgWidth = image.shape
characters = [] 
for c in contours:
	x, y, w, h = cv2.boundingRect(c)
	if (pow(max(h,w),2)>(imgHeight*imgWidth)/1200 and pow(max(h,w),2)<(imgHeight*imgWidth)/5):
		img = binImg[y:y+h,x:x+w]
		char = Character(img,x,y,w,h)
		characters.append(char)
		
imgWithBounds = ip.drawCharacterBounds(binImg.copy(), characters)
cv2.imwrite("figures/preCharProcessing.jpg", imgWithBounds)

cp.processCharacters(characters)
imgWithBounds = ip.drawCharacterBounds(binImg.copy(), characters)
cv2.imwrite("figures/postCharProcessing.jpg", imgWithBounds)

cv2.imwrite("figures/characterPreFormatting.jpg", characters[0].image)

for char in characters:
	ip.squareImage(char)
	ip.resize(char,(45,45))		
		
cv2.imwrite("figures/characterPostFormatting.jpg", characters[0].image)



classifiedImg = ip.drawClassifiedCharacters(imgWithBounds, classifier, characters)
cv2.imwrite("figures/classifiedCharacters.jpg", classifiedImg)

lines = parseEquation(characters)
final = binImg.copy()
drawSolvedEquations(lines,final, 'x')
cv2.imwrite("figures/solvedEquation.jpg", final)
