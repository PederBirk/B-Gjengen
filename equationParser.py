from math import fabs, floor


def getCenter(char):
	return(char.xPos + char.width/2, char.yPos + char.height/2)
	
def getAvgCharSize(chars):
	l = len(chars)
	if l == 0:
		return (0, 0)
	xSum, ySum = 0, 0
	for c in chars:
		xSum += c.width
		ySum += c.height
	return (xSum/l, ySum/l)

def sortCharsByX(chars):
	chars.sort(key=lambda x:x.xPos+x.width/2)
	return chars
	
def sortCharsByY(chars):
	chars.sort(key=lambda x:x.yPos+x.height/2)
	return chars
	
def getLines(chars, deltaY):
	chars = sortCharsByY(chars)
	ret = [[chars[0]]]
	for i in range(len(chars)-1):
		thisX, thisY = getCenter(chars[i])
		nextX, nextY = getCenter(chars[i+1])
		if nextY > thisY + deltaY:
			ret.append([])
		ret[-1].append(chars[i+1])
	return ret

def groupCharsInLine(chars, deltaX):
	sortCharsByX(chars)
	ret = [[chars[0]]]
	for i in range(len(chars)-1):
		thisX = chars[i].xPos + chars[i].width
		nextX = chars[i+1].xPos
		if nextX > thisX + deltaX:
			ret.append([])
		ret[-1].append(chars[i+1])
	return ret
	
def parseEquation(chars):
	avgWidth, avgHeight = getAvgCharSize(chars)
	lines = getLines(chars, avgHeight/2)
	lines = [groupCharsInLine(line, avgWidth*1.5) for line in lines]
	ret = []
	'''
	for line in lines:
		for group in line:
			ret.append(((group[0].xPos, group[0].yPos),(group[-1].xPos+group[-1].width, group[-1].yPos+group[-1].height)))
	'''
	return lines
			