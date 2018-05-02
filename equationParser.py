from sympy import Symbol
from sympy import solve
from sympy.solvers.solveset import linsolve
from sympy.parsing.sympy_parser import parse_expr
import cv2

#Get center position of character bounding box
def getCenter(char):
	return(char.xPos + char.width/2, char.yPos + char.height/2)

#Get the average size of the character boudning boxes	
def getAvgCharSize(chars):
	l = len(chars)
	if l == 0:
		return (0, 0)
	xSum, ySum = 0, 0
	for c in chars:
		xSum += c.width
		ySum += c.height
	return (xSum/l, ySum/l)

#Sort characters by the x position of the center of the bounding box
def sortCharsByX(chars):
	chars.sort(key=lambda x:x.xPos+x.width/2)
	return chars

#Sort characters by the y position of the center of the bounding box	
def sortCharsByY(chars):
	chars.sort(key=lambda x:x.yPos+x.height/2)
	return chars

#Group characters who are on the same line
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

#Group adjecent characters in a line
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

#Group all characters in image into individual equations	
def parseEquation(chars):
	avgWidth, avgHeight = getAvgCharSize(chars)
	lines = getLines(chars, avgHeight/2)
	lines = [groupCharsInLine(line, avgWidth*3) for line in lines]
	return lines

#Prepare equation to be input to SymPy 
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

#Solve a single equation with respect to the variable specified in the arguments		
def solveSingleEq(eq, var):
	sym = Symbol(var)
	sol = solve(eq,sym)
	print(var + " = " + str(sol[0]))
	return sol

#Solve a system of two eqautions, where x and y are the unknown	
def solveSystemOfEqs(eqs):
	x = Symbol("x")
	y = Symbol("y")
	sol = linsolve([parse_expr(eqs[0],evaluate=0), parse_expr(eqs[1], evaluate=0)],(x,y))
	x_sol,y_sol = next(iter(sol))
	print ("x = " + str(x_sol) + ", y = " + str(y_sol))
	return (x_sol, y_sol)

#Find the boudning box around an equation
def findEquationBoundingBox(group):
	x_min = group[0].xPos
	x_max = group[-1].xPos + group[-1].width
	y_min = 1000000
	y_max = -y_min
	for char in group:
		y_min = min(y_min,char.yPos)
		y_max = max(y_max,char.yPos + char.height)
	
	return(x_min,y_min,x_max,y_max)

#Draw the bounding box and solution for all equation which can be solved in an image	
def drawSolvedEquations(lines, image, withRespectTo):
	sym = Symbol(withRespectTo)
	eqs = toSymPyFormat(lines)
	i = 0
	for line in lines:
		for group in line:
			try:
				sol = solve(eqs[i],sym)
				x_min,y_min,x_max,y_max = findEquationBoundingBox(group)
				cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0, 0, 0))
				string = withRespectTo + " = " + str(sol[0])
				cv2.putText(image, string, (x_min+50, y_max+20), cv2.FONT_HERSHEY_PLAIN, 1.5, 2)
				i += 1
			except Exception:			
				i += 1
			