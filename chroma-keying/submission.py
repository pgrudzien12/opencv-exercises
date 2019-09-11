import cv2
import numpy as np
from datetime import datetime

import greenscreen
import kj
import smoothstep
import ChromaToTransparency
import first

toleranceFactor = 10
softnessFactor = 10
defringeFactor = 100
green = 60

windowName = "Video"
frame = None
zoom = True
changed = True
play = False
readOne = True

# load an image
#cap = cv2.VideoCapture("greenscreen-asteroid.mp4")
cap = cv2.VideoCapture("greenscreen-demo.mp4")
#cap = cv2.VideoCapture("destroyer.mp4")
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
capLength = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT)

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def onMouse(action, x, y, flags, userdata):
    global green, frame, changed
    if action==cv2.EVENT_LBUTTONDOWN:
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = np.median(image_hsv[y-3:y+3,x-3:x+3,0])
        changed = True
    if action==cv2.EVENT_RBUTTONDOWN:
        patch = frame[y-40:y+40,x-40:x+40]
        cv2.imshow("patch", cv2.resize(patch, (400,400)))
    return



def onChangeTolerance():
    global toleranceFactor, changed
    toleranceFactor = cv2.getTrackbarPos("Tolerance", windowName)
    changed = True

def onChangeSoftness():
    global softnessFactor, changed
    softnessFactor = cv2.getTrackbarPos("Softness", windowName)
    changed = True

def onChangeDefringe():
    global defringeFactor, changed
    defringeFactor = cv2.getTrackbarPos("Defringe", windowName)
    changed = True
    
def onPositionChange(*args):
    global cap, capLength, readOne, changed
    progress = cv2.getTrackbarPos("Progress", windowName)
    progress = int((progress /100) * capLength)
    cap.set(cv2.CAP_PROP_POS_FRAMES, progress)
    readOne = True
    changed = True

cv2.setMouseCallback(windowName, onMouse)

while(cap.isOpened()):
    # Capture frame-by-frame
    if play or readOne:
        ret, frame = cap.read()
        if ret == True:
            resultFrame = frame
            if zoom:
                resultFrame = frame[500:600,620:700,:]
        readOne = False
    key = cv2.waitKey(15)
    if key & 0xFF == 27:
        break
    
    if key & 0xFF == 32:
        play = not play

    if key & 0xFF == 122:
        zoom = not zoom
        play = False

    # Display the resulting frame
    if not changed and not play:
        continue
    #processed = kj.kj(resultFrame, [toleranceFactor, softnessFactor, defringeFactor, green])
    processed = process(resultFrame, [toleranceFactor, softnessFactor, defringeFactor, green])
    #processed = smoothstep.chroma_keying(resultFrame.copy(), 
        np.array((0, 255, 0)), 
        toleranceFactor/100, 
        (softnessFactor+toleranceFactor)/100)
    if zoom:
        processed = cv2.resize(processed, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(windowName, processed)
    cv2.setWindowTitle(windowName, "Image Green Hue={0}".format(green))

    # display progress trackbar
    cFrameNo = cap.get(cv2.CAP_PROP_POS_FRAMES); 
    cv2.createTrackbar("Progress", windowName, int(100 * cFrameNo / capLength) , 100, onPositionChange)

    # other trackbars
    cv2.createTrackbar("Tolerance", windowName, toleranceFactor, 100, lambda *args: onChangeTolerance())
    cv2.createTrackbar("Softness", windowName, softnessFactor, 100, lambda *args: onChangeSoftness())
    cv2.createTrackbar("Defringe", windowName, defringeFactor, 100, lambda *args: onChangeDefringe())

cv2.destroyAllWindows()