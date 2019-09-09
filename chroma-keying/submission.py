import cv2
import numpy as np
from datetime import datetime

toleranceFactor = 10
softnessFactor = 10
defringeFactor = 100
green = 60

windowName = "Video"
frame = None

# load an image
#cap = cv2.VideoCapture("greenscreen-asteroid.mp4")
cap = cv2.VideoCapture("greenscreen-demo.mp4")
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
capLength = int(cv2.VideoCapture.get(cap, property_id))

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def onMouse(action, x, y, flags, userdata):
    global green, frame
    if action==cv2.EVENT_LBUTTONDOWN:
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = np.median(image_hsv[y-3:y+3,x-3:x+3,0])
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

def defringe(image, mask, defringeFactor):
    # pivot points for X-Coordinates
    originalValue = np.array([0, 50, 100, 150, 200, 255])
    factor = (defringeFactor) / 100
    # Changed points on Y-axis for each channel
    gCurve = originalValue * factor

    # Create a LookUp Table
    fullRange = np.arange(0,256)
    gLUT = np.interp(fullRange, originalValue, gCurve)

    # Get the red channel and apply the mapping
    gChannel = image[:,:,1]
    gChannel = cv2.LUT(gChannel, gLUT)
    image[:,:,1] = gChannel

    return image

def blurMask(mask, softnessFactor):
    ksize = softnessFactor // 2
    if (ksize % 2) == 0:
        ksize = ksize + 1

    size = (ksize, ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    morphed = mask.copy()
    #morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    #morphed = cv2.erode(morphed, kernel)
    #morphed = cv2.GaussianBlur(morphed, size, 2)
    return morphed


def getMaskEdges(mask):
    lowThreshold = 100
    ratio = 3
    kernelSize = 3
    size = (31, 31)
    edges = cv2.Canny(mask, lowThreshold, lowThreshold * ratio, kernelSize);
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    edges = cv2.dilate(edges, kernel, iterations=2)
    cv2.imshow("edges", edges)

    return None

def preprocess(image):
    return cv2.GaussianBlur(image, (3,3), 2)

def process(image_original):
    global toleranceFactor, softnessFactor, defringeFactor, green
    image = image_original
#    image = preprocess(image_original)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = buildToleranceMask(image_hsv, toleranceFactor, green)
    mask_edges = getMaskEdges(mask)
    #mask = blurMask(mask, softnessFactor)

    image_hsv[:,:,2] = np.bitwise_and(image_hsv[:,:,2], mask)

    color_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    
    return defringe(color_image, mask, defringeFactor)

def onChangeTolerance():
    global toleranceFactor
    toleranceFactor = cv2.getTrackbarPos("Tolerance", windowName)

def onChangeSoftness():
    global softnessFactor
    softnessFactor = cv2.getTrackbarPos("Softness", windowName)

def onChangeDefringe():
    global defringeFactor
    defringeFactor = cv2.getTrackbarPos("Defringe", windowName)
    
def onPositionChange(*args):
    global cap, capLength, readOne
    progress = cv2.getTrackbarPos("Progress", windowName)
    progress = int((progress /100) * capLength)
    cap.set(cv2.CAP_PROP_POS_FRAMES, progress)
    readOne = True

play = True
readOne = False
cv2.setMouseCallback(windowName, onMouse)

while(cap.isOpened()):
    # Capture frame-by-frame
    if play or readOne:
        ret, frame = cap.read()
        if ret == True:
            resultFrame = frame
        readOne = False
    key = cv2.waitKey(15)
    if key & 0xFF == 27:
        break
    
    if key & 0xFF == 32:
        play = not play

    # Display the resulting frame
    cv2.imshow(windowName, process(resultFrame))
    cv2.setWindowTitle(windowName, "Image Green Hue={0}".format(green))

    # display progress trackbar
    cFrameNo = cap.get(cv2.CAP_PROP_POS_FRAMES); 
    cv2.createTrackbar("Progress", windowName, int(100 * cFrameNo / capLength) , 100, onPositionChange)

    # other trackbars
    cv2.createTrackbar("Tolerance", windowName, toleranceFactor, 100, lambda *args: onChangeTolerance())
    cv2.createTrackbar("Softness", windowName, softnessFactor, 100, lambda *args: onChangeSoftness())
    cv2.createTrackbar("Defringe", windowName, defringeFactor, 100, lambda *args: onChangeDefringe())

cv2.destroyAllWindows()