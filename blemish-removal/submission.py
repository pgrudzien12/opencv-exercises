# Enter your code here
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def onMouse(action, x, y, flags, userdata): 
    if action==cv2.EVENT_LBUTTONDOWN:
        center=[(x,y)]
        # Mark the center
        patch = userdata[y-15:y+15,x-15:x+15]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = cv2.Laplacian(patch, -1)
        patch = cv2.resize(patch, (200,200))
        cv2.setWindowTitle("patch", "lapl")
        cv2.imshow("patch", patch)


source = cv2.imread("blemish.png",1)
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()
cv2.namedWindow("Window")
cv2.namedWindow("patch")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", onMouse, source)
k = 0
# loop until escape character is pressed
while k!=27 :
  cv2.imshow("Window", source)
  k = cv2.waitKey(20) & 0xFF
  # Another way of cloning
  if k==99:
    source= dummy.copy()


cv2.destroyAllWindows()
