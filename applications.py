import cv2
import imageProcessor as ip
from keras.models import model_from_json
from equationParser import parseEquation
from equationParser import toSymPyFormat
from equationParser import solveSystemOfEqs
from equationParser import solveSingleEq
from equationParser import drawSolvedEquations
from ClassifierKeras import ClassifierKeras

#Draw character bounds and classify characters from a webcam stream
def classifyCharsFromWebcam():
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
	
	json_file = open('models/model_balanced.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/model_balanced.h5")
	
	classifier = ClassifierKeras(symbols, loaded_model)
	
	cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
	vc = cv2.VideoCapture(0)
	
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False
	
	while rval:
		cv2.resizeWindow("preview", 640, 480)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # ESC
			break
		
		#Process frame
		cv2.imwrite("screencap.jpg", frame)
		threshed_img = ip.getImgMat("screencap.jpg")	
		characters = ip.extractCharacters(threshed_img, (45,45))
		imgWithBounds = ip.drawCharacterBounds(frame, characters)
		classifiedImg = ip.drawClassifiedCharacters(imgWithBounds, classifier, characters)
		cv2.imshow("preview", classifiedImg)
	
	cv2.destroyWindow("preview")
	

#Solve a system of equations from input image
def solveSystemFromImage():
	img = ip.getImgMat('images/twoEq.jpg')
	chars = ip.extractCharacters(img, (45,45))
	lines = parseEquation(chars)
	
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
	
	json_file = open('models/model_balanced.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/model_balanced.h5")
	
	classifier = ClassifierKeras(symbols, loaded_model)
	
	print("Equations:\n")
	
	for line in lines:
		for group in line:
			str = ""
			for c in group:
				classifier.classify(c)
				str += c.symbol
			print(str, end="\t")
		print("")
	
	eqs = toSymPyFormat(lines)
	
	print("\nSolution:\n")	
	
	solveSystemOfEqs(eqs)
	
	img = cv2.pyrDown(img)
	cv2.imshow("test", img)
	 
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

#Solve individual equation from input image	
def solveEqFromImage(withRespectTo):
	img = ip.getImgMat('images/twoEq.jpg')
	chars = ip.extractCharacters(img, (45,45))
	lines = parseEquation(chars)
	
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
	
	json_file = open('models/model_balanced.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/model_balanced.h5")
	
	classifier = ClassifierKeras(symbols, loaded_model)
	
	print("Equations:\n")
	
	for line in lines:
		for group in line:
			str = ""
			for c in group:
				classifier.classify(c)
				str += c.symbol
			print(str, end="\t")
		print("")
	
	print("\nSolutions:\n")
	
	eqs = toSymPyFormat(lines)	
	for eq in eqs:
		solveSingleEq(eq,withRespectTo)
	
	img = cv2.pyrDown(img)
	cv2.imshow("test", img)
	 
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

#Draw character bounds and classify characters from input image
def classifyCharsFromImage():
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
	
	json_file = open('models/model_balanced.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/model_balanced.h5")
	
	classifier = ClassifierKeras(symbols,loaded_model)
	
	imgPath = 'images/twoEq.jpg'
	img = ip.getImgMat(imgPath)
	
	chars = ip.extractCharacters(img)
	imgRect = img.copy()
	ip.drawCharacterBounds(imgRect,chars)
	ip.drawClassifiedCharacters(imgRect, classifier, chars)
	cv2.imshow("Classifications", imgRect)
	if cv2.waitKey(0) & 0xff == 27:
		 cv2.destroyAllWindows()

#Solve all equation in webcam stream, draw bounding box and solution on the images		 
def solveEqsFromWebcam(withRespectTo):
	symbols = ['0','1','2','3','4','5','6','7','8','9', '=', 'x', '+', '-', 'y']
	
	json_file = open('models/model_balanced.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("models/model_balanced.h5")
	
	classifier = ClassifierKeras(symbols, loaded_model)
	
	cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
	vc = cv2.VideoCapture(0)
	
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False
	
	while rval:
		cv2.resizeWindow("preview", 640, 480)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # ESC
			break
		
		# treat frame
		cv2.imwrite("screencap.jpg", frame)
		img = ip.getImgMat("screencap.jpg")
		chars = ip.extractCharacters(img, (45,45))
		
		for char in chars:
			classifier.classify(char)
		
		lines = parseEquation(chars)
		
		drawSolvedEquations(lines, frame, withRespectTo)
		cv2.imshow("preview", frame)
	
	cv2.destroyWindow("preview")