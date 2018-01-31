import cv2
import numpy as np 

img = cv2.pyrDown(cv2.imread('pic.jpg', cv2.IMREAD_UNCHANGED))
 
# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)

# find contours and get the external one
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a rectangle to visualize the bounding rect
    cv2.rectangle(threshed_img, (x, y), (x+w, y+h), (0, 0, 0))
	
cv2.imshow("contours", threshed_img)
 
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()