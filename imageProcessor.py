import cv2
import Character as ch
import numpy as np
import processCharacters as pc

def getImgMat(path, thresh_lower = 180, thresh_upper = 255):
	img = cv2.pyrDown(cv2.imread(path, cv2.IMREAD_UNCHANGED))
	kernel = np.ones((7,7),np.float32)/49
	img = cv2.filter2D(img,-1,kernel)
	threshed_image = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,41,6)
	return threshed_image

def extractCharacters(binImg, targetResolution = (25,25), thresh_lower = 180, thresh_upper = 255):
	image, contours, hier = cv2.findContours(binImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	characters = [] 
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		if (h+w>25 and h+w<200): #disregard tiny and huge components
			img = binImg[y:y+h,x:x+w]
			char = ch.Character(img,x,y,w,h)
			characters.append(char)
			
	pc.processCharacters(characters)
	for char in characters:
		squareImage(char)
		resize(char,targetResolution)
	
	return characters

def squareImage(char):
	w = char.width
	h = char.height
	if w > h:
		diff = w-h
		padding = int(diff/2.0)
		dim = (padding,w)
		paddingMat = 255*np.ones(dim,dtype=np.uint8)
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
		
def resize(char,targetResolution):
	char.image = cv2.resize(char.image,targetResolution)
	ret, char.image = cv2.threshold(char.image, 128, 255, cv2.THRESH_BINARY)
	
def webcam():
	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)
	
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False

	while rval:
		cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # ESC
			break
	
	cv2.destroyWindow("preview")
