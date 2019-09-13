import cv2
import numpy as np
from datetime import datetime

slopeFactor = 20
thresholdFactor = 80
edgesFactor = 50
green = np.array((0,255,0))

windowName = "Video"
frame = None
zoom = False
changed = True
play = True
readOne = True

# load an image
cap = cv2.VideoCapture("greenscreen-asteroid.mp4")
#cap = cv2.VideoCapture("greenscreen-demo.mp4")

if (cap.isOpened()== False): 
    print("Error opening video stream or file")
capLength = cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT)

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def onMouse(action, x, y, flags, userdata):
    global green, frame, changed
    if action==cv2.EVENT_LBUTTONDOWN:
        #green = np.array((0, frame[y,x,1], 0))
        changed = True
    if action==cv2.EVENT_RBUTTONDOWN:
        patch = frame[y-40:y+40,x-40:x+40]
        cv2.imshow("patch", cv2.resize(patch, (400,400)))
    return



def onChangeSlope():
    global slopeFactor, changed
    slopeFactor = cv2.getTrackbarPos("Slope", windowName)
    changed = True

def onChangeThreshold():
    global thresholdFactor, changed
    thresholdFactor = cv2.getTrackbarPos("Threshold", windowName)
    changed = True

def onChangeEdges():
    global edgesFactor, changed
    edgesFactor = cv2.getTrackbarPos("Edges", windowName)
    changed = True
    
def onPositionChange(*args):
    global cap, capLength, readOne, changed
    progress = cv2.getTrackbarPos("Progress", windowName)
    progress = int((progress /100) * capLength)
    cap.set(cv2.CAP_PROP_POS_FRAMES, progress)
    readOne = True
    changed = True

cv2.setMouseCallback(windowName, onMouse)

def getMaskEdges(mask):
    lowThreshold = 100
    ratio = 3
    kernelSize = 3
    size = (5, 5)
    edges = cv2.Canny(np.uint8(mask*255), lowThreshold, lowThreshold * ratio, kernelSize);
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def removeGreenUsingLUT(image, mask, factor):
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    # Changed points on Y-axis for each channel
    gCurve = originalValue * factor

    # Create a LookUp Table
    fullRange = np.arange(0,256)
    gLUT = np.interp(fullRange, originalValue, gCurve)

    # Get the red channel and apply the mapping
    gChannel = cv2.bitwise_and(image[:,:,1], mask)
    gChannel = np.uint8(cv2.LUT(gChannel, gLUT))
    return gChannel

def removeGreenBruteforce(image, mask, factor):
    r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
    g[mask > 0] = np.uint8(factor * (r+b))[mask > 0]
    return g

def correctEdges(image, mask, edges):
    mask_edges = getMaskEdges(mask)
    #gChannel = removeGreenUsingLUT(image, mask_edges, edges / 100)
    gChannel = removeGreenBruteforce(image, mask_edges, edges / 100 )
    #gChannel = cv2.blur(gChannel, (3,3))

    if zoom:
        processed = cv2.resize(gChannel, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
        processed[0, 0] = 255
        cv2.imshow("LUT gChannel", processed)
    #green_channel = image[:,:,1]
    #image[:,:,1] = gChannel
    if zoom:
        processed = cv2.resize(green_channel, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("green_channel", processed)

    return image

def smoothstep( edge0,  edge1, image):
  x = np.clip((image - edge0) / (edge1 - edge0), 0.0, 1.0); 
  sq = np.multiply(x, x)
  return np.multiply(sq, (3 - 2 * x))

def chroma_keying(image, color, slope, threshold):
    normedIm = image/255
    normedColor = color/255
    allGreen = np.full_like(normedIm, normedColor)
    d = np.linalg.norm(allGreen - normedIm, axis=2)
    edge0 = threshold * (1.0 - slope)
    alpha = smoothstep(edge0, threshold, d)
    image[:,:,0] = np.multiply(image[:,:,0], alpha)
    image[:,:,1] = np.multiply(image[:,:,1], alpha)
    image[:,:,2] = np.multiply(image[:,:,2], alpha)

    return image, alpha


def process(image, params):
    slope, threshold, edgesFactor, green = params
    image, alpha = chroma_keying(image, green, slope/ 100, threshold / 100)

    return correctEdges(image, alpha, edgesFactor)

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
    #processed = kj.kj(resultFrame, [slopeFactor, thresholdFactor, edgesFactor, green])
    processed = process(resultFrame.copy(), [slopeFactor, thresholdFactor, edgesFactor, green])
    #processed = chroma_keying(resultFrame.copy(), 
    #    green, 
    #    slopeFactor / 100, 
    #    thresholdFactor / 100)
    if zoom:
        processed = cv2.resize(processed, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(windowName, processed)
    cv2.setWindowTitle(windowName, "Image Green Hue={0}".format(green))

    # display progress trackbar
    cFrameNo = cap.get(cv2.CAP_PROP_POS_FRAMES); 
    cv2.createTrackbar("Progress", windowName, int(100 * cFrameNo / capLength) , 100, onPositionChange)

    # other trackbars
    cv2.createTrackbar("Slope", windowName, slopeFactor, 100, lambda *args: onChangeSlope())
    cv2.createTrackbar("Threshold", windowName, thresholdFactor, 100, lambda *args: onChangeThreshold())
    cv2.createTrackbar("Edges", windowName, edgesFactor, 100, lambda *args: onChangeEdges())

cv2.destroyAllWindows()