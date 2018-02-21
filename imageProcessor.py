import cv2
import Character as ch
import numpy as np 

def getBinaryImg(path, tresh_lower = 180, tresh_upper = 255):
	img = cv2.pyrDown(cv2.imread('images/pic.jpg', cv2.IMREAD_UNCHANGED))
	ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)
	return threshed_img

def extractCharacters(binImg):
	image, contours, hier = cv2.findContours(binImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	characters = [] 
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		if (h+w>25 and h+w<200): #disregard tiny and huge components
			img = binImg[y:y+h,x:x+w]
			char = ch.Character(img,x,y,w,h)
			characters.append(char)
	return characters

