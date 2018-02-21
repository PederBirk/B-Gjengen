import cv2
import numpy as np 

#Takes one image and splits into individual components. Returns list of these images in binary.
def split(img):
 
	# threshold image
	ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY_INV)

	# find contours and get the external one
	image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	 
	characters = []
	 
	for c in contours:
		# get the bounding rect
		x, y, w, h = cv2.boundingRect(c)
		# split original image into new images containing each character
		if(h+w>25 and h+w<200): #disregard tiny and huge components
			characters.append(threshed_img[y:y+h,x:x+w]) #extract the relevant areas of the image
	return characters