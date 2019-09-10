import cv2
import numpy as np
from datetime import datetime

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

def buildToleranceMask(img_hsv, toleranceFactor, green):
    hue = img_hsv[:,:,0]
    saturation = img_hsv[:,:,1]
    value = img_hsv[:,:,2]
    tolerance = (toleranceFactor / 100) * green
    lb_green, hb_green = green - tolerance, green + tolerance
    hue_tolerance = np.logical_and(lb_green <= hue, hue <= hb_green)
    saturated = np.logical_and(hue_tolerance, saturation >= 60)
    valued = np.logical_and(saturated, value >= 10)
    value_mask = np.ones_like(hue) * 255
    value_mask[valued] = 0
    return value_mask

def removeGreenUsingLUT(image, mask, factor):
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    # Changed points on Y-axis for each channel
    gCurve = originalValue * factor
    gCurve[1] = 50
    gCurve[2] = 100

    # Create a LookUp Table
    fullRange = np.arange(0,256)
    gLUT = np.interp(fullRange, originalValue, gCurve)

    # Get the red channel and apply the mapping
    gChannel = cv2.bitwise_and(image[:,:,1], mask)
    gChannel = np.uint8(cv2.LUT(gChannel, gLUT))
    return gChannel

def removeGreenBruteforce(image, mask, factor):
    green_channel = image[:,:,1].copy()
    green_channel = np.uint8(green_channel * factor)
    return green_channel


def defringe(image, mask, defringeFactor):
    mask_edges = getMaskEdges(mask)
    #gChannel = removeGreenUsingLUT(image, mask_edges, defringeFactor / 100)
    gChannel = removeGreenBruteforce(image, mask_edges, defringeFactor / 100)
    #gChannel = cv2.blur(gChannel, (3,3))

    if zoom:
        processed = cv2.resize(gChannel, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
        processed[0, 0] = 255
        cv2.imshow("LUT gChannel", processed)
    green_channel = image[:,:,1]
    #green_channel[mask_edges > 0] = gChannel[mask_edges > 0]
    image[:,:,1] = gChannel
    if zoom:
        processed = cv2.resize(green_channel, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("green_channel", processed)

    return image

def blurMask(mask, softnessFactor):
    ksize = softnessFactor // 2
    if (ksize % 2) == 0:
        ksize = ksize + 1

    size = (ksize, ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    morphed = mask.copy()
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    #morphed = cv2.erode(morphed, kernel)
    #morphed = cv2.GaussianBlur(morphed, size, 2)
    #processed = cv2.resize(morphed, (-1,-1), fx = 5, fy = 5, interpolation=cv2.INTER_NEAREST)
    #cv2.imshow("mask", processed)
    return morphed


def getMaskEdges(mask):
    lowThreshold = 100
    ratio = 3
    kernelSize = 3
    size = (5, 5)
    edges = cv2.Canny(mask, lowThreshold, lowThreshold * ratio, kernelSize);
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def preprocess(image):
    return cv2.GaussianBlur(image, (3,3), 2)

def process(image_original):
    global toleranceFactor, softnessFactor, defringeFactor, green
    image = image_original
#    image = preprocess(image_original)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = buildToleranceMask(image_hsv, toleranceFactor, green)
    mask = blurMask(mask, softnessFactor)

    image_hsv[:,:,2] = np.bitwise_and(image_hsv[:,:,2], mask)

    color_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    
    return defringe(color_image, mask, defringeFactor)

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
    processed = process(resultFrame)
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