import cv2
import numpy as np

#Takes binary image and returns skeletized version
def skeletize(img):
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
			return skeleton