import split
import processCharacters
import numpy as np
import cv2

image = cv2.imread('pic2.jpg')

cv2.imshow('hello world',image)
cv2.waitKey(0)
chars = split.split(image)

for img in chars:
	img.show()
	
chars = processCharacters.processCharacters(chars)

for img in chars:
	img.show()

