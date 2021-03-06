import numpy as np
import Character

#Check if the bounding boxes of two charactes intersect
def charIntersectes(a, b):
	if(a.xPos+a.width/2 > b.xPos) and (a.xPos+a.width/2 < b.xPos+b.width):
		if(a.yPos+a.width/2 > b.yPos) and (a.yPos+a.width/2 < b.yPos+b.height):
			return True
	return False

#Check if two characters are directly above each other	
def charsAbove(a, b):
	if (-a.xPos + b.xPos) < (a.width+b.width)/4 :
		#Over hverandre
		if(abs(a.yPos - b.yPos)) <= (a.width+b.width)/2:
			return True
	return False

#Sort characters by x-position	
def sortChars(chars):
	chars.sort(key=lambda x:x.xPos)
	return chars

#Merge two characters	
def combine(a, b):
	xMax = max(a.xPos+a.width, b.xPos+b.width)
	xMin = min(a.xPos, b.xPos)
	yMax = max(a.yPos+a.height, b.yPos+b.height)
	yMin = min(a.yPos, b.yPos)
	
	char = np.ones((yMax-yMin, xMax-xMin),dtype=np.uint8)*255
	
	char[a.yPos-yMin:a.yPos+a.height-yMin, a.xPos-xMin:a.xPos+a.width-xMin] = a.image
	char[b.yPos-yMin:b.yPos+b.height-yMin, b.xPos-xMin:b.xPos+b.width-xMin] = b.image
	
	return Character.Character(char, xMin, yMin, xMax-xMin, yMax-yMin)
	

#Merge all characters in image which intersects or are above
def processCharacters(chars):
	i = 0
	chars = sortChars(chars)
	while i < len(chars)-1:
		shouldCombine = False
		if(charIntersectes(chars[i], chars[i+1]) or charIntersectes(chars[i+1], chars[i])):
			shouldCombine = True
		
		if(charsAbove(chars[i], chars[i+1])):
			shouldCombine = True
		
		if(shouldCombine): 
			chars[i] = combine(chars[i], chars[i+1])
			chars.pop(i+1)
		else:
			i+=1
	return chars