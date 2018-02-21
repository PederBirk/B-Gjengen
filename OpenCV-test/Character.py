import cv2
import numpy as np

class Character:

	symbol = None #The symbol the neural network has identified
	   
	def __init__(self,image,xPos,yPos,width,height):
		self.image = image #Image matrix to input to Neural Network
		self.xPos = xPos #x position of top left corner of bounding box in image
		self.yPos = yPos #y position of top left corner of bounding box in image
		self.width = width #width of bounding box
		self.height = height #height of bounding box
		
	def show(self):
		cv2.imshow("Character", self.image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		