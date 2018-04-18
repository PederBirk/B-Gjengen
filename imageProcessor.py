import cv2
import Character as ch
import numpy as np
import characterProcessor as cp
from math import pow

#Read and threshhold image
def getImgMat(path, thresh_lower = 180, thresh_upper = 255):
	img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
	threshed_image = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,41,6)
	return threshed_image

#Extract individual characters from image
def extractCharacters(binImg, targetResolution = (45,45), thresh_lower = 180, thresh_upper = 255):
	image, contours, hier = cv2.findContours(binImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	imgHeight, imgWidth = image.shape
	characters = [] 
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		if (pow(max(h,w),2)>(imgHeight*imgWidth)/1200 and pow(max(h,w),2)<(imgHeight*imgWidth)/5): #TODO improve; disregard tiny and huge components
			img = binImg[y:y+h,x:x+w]
			char = ch.Character(img,x,y,w,h)
			characters.append(char)
			
	cp.processCharacters(characters)
	for char in characters:
		squareImage(char)
		resize(char,targetResolution)		
	
	return characters

#Make image square by adding white space in the shortest direction
def squareImage(char):
	w = char.width
	h = char.height
	if w > h:
		diff = w-h
		padding = int(diff/2.0)
		dim = (padding,w)
		paddingMat = 225*np.ones(dim,dtype=np.uint8)
		char.image = np.concatenate((paddingMat,char.image,paddingMat),0)
	elif h > w:
		diff = h-w
		padding = int(diff/2.0)
		dim = (h,padding)
		paddingMat = 255*np.ones(dim,dtype=np.uint8)
		char.image = np.concatenate((paddingMat,char.image,paddingMat),1)
		
	size = char.image.shape
	if size[0] > size[1]:
		char.image = np.concatenate((char.image,255*np.ones((h,1),dtype=np.uint8)),1)
	elif size[1] > size[0]:
		char.image = np.concatenate((char.image,255*np.ones((1,w),dtype=np.uint8)),0)

#Resize image		
def resize(char,targetResolution):
	char.image = cv2.resize(char.image,targetResolution)
	ret, char.image = cv2.threshold(char.image, 128, 255, cv2.THRESH_BINARY)

#Draw bounding boxes around each character
def drawCharacterBounds(image, characters):
	for c in characters:
		cv2.rectangle(image, (c.xPos, c.yPos), (c.xPos + c.width, c.yPos + c.height), (0, 0, 0))
	return image

#Draw the classified character above each box in image
def drawClassifiedCharacters(image, classifier, characters):
	for c in characters:
		classifier.classify(c)
		cv2.putText(image, c.symbol, (c.xPos, c.yPos), cv2.FONT_HERSHEY_PLAIN, 2, 1)
	return image
	
#Takes binary image and returns skeletized version
def skeletize(img):
	img = cv2.bitwise_not(img)
	skeleton = np.zeros(img.shape,np.uint8)
	eroded = np.zeros(img.shape,np.uint8)
	temp = np.zeros(img.shape,np.uint8)
	thresh = img
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	
	while(True):
		cv2.erode(thresh, kernel, eroded)
		cv2.dilate(eroded, kernel, temp)
		cv2.subtract(thresh, temp, temp)
		cv2.bitwise_or(skeleton, temp, skeleton)
		thresh, eroded = eroded, thresh # Swap instead of copy

		if cv2.countNonZero(thresh) == 0:
			return cv2.bitwise_not(skeleton)