# Enter your code here
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def calcVariance(patch):
        x = cv2.Scharr(patch, -1, 1, 0)
        y = cv2.Scharr(patch, -1, 0, 1)

        return np.abs(x) + np.abs(y)

def onMouse(action, x, y, flags, userdata): 
    if action==cv2.EVENT_LBUTTONDOWN:
        # Mark the center
        patch = userdata[y-15:y+15,x-15:x+15]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:,:,0]
        variance = calcVariance(patch)
        size = (400,400)
        resizedVariance = cv2.resize(variance, size)
        resizedPatch = cv2.resize(patch, size)
        cv2.setWindowTitle("patch", "{0}".format(np.sum(variance)))
        cv2.imshow("patch", np.hstack([resizedVariance, resizedPatch]))

source = cv2.imread("blemish-removal/blemish.png",1)
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
