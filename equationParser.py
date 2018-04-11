from math import fabs, floor
from sympy import Symbol
from sympy import solve


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
	lines = [groupCharsInLine(line, avgWidth*3) for line in lines]
	ret = []
	'''
	for line in lines:
		for group in line:
			ret.append(((group[0].xPos, group[0].yPos),(group[-1].xPos+group[-1].width, group[-1].yPos+group[-1].height)))
	'''
	return lines

def toSymPyFormat(lines):
	eqs = []
	for line in lines:
		for group in line:
			eq = group[0].symbol
			for i in range(1,len(group)):
				if not (group[i].symbol == "+" or group[i].symbol == "-" or group[i].symbol == "=" or group[i-1].symbol == "+" or group[i-1].symbol == "-" or group[i-1].symbol == "="):
					 eq += "*"
					 eq += group[i].symbol
				elif group[i].symbol == "=":
					eq += "-("
				else:
					eq += group[i].symbol
			eq += ")"
			eqs.append(eq)
	return eqs
			
def solveSingleEq(eq):
	x = Symbol("x")
	y = Symbol("y")
	if "x" in eq:
		sol = solve(eq,x)
		var = "x"
	elif "y" in eq:
		sol = solve(eq,y)
		var = "x"
	else:
		print("Equation does not contain x or y!")
	print(var + " = " + str(sol[0]))
		
	
			