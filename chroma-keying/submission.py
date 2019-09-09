import cv2
import numpy as np

toleranceFactor = 10
softnessFactor = 10
defringeFactor = 10

windowName = "Video"

# load an image
cap = cv2.VideoCapture("greenscreen-asteroid.mp4")
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
capLength = int(cv2.VideoCapture.get(cap, property_id))

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def buildToleranceMask(img_hsv, toleranceFactor, green):
    hue = img_hsv[:,:,0]
    tolerance = (toleranceFactor / 100)* green
    lb_green, hb_green = green - tolerance, green + tolerance
    hue_tolerance = np.logical_and(lb_green <= hue, hue <= hb_green)
    value_mask = np.ones_like(hue) * 255
    value_mask[hue_tolerance] = 0
    return value_mask


def blurMask(mask, softnessFactor):
    ksize = softnessFactor // 2
    if (ksize % 2) == 0:
        ksize = ksize + 1

    size = (ksize, ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.erode(morphed, kernel)
    return cv2.GaussianBlur(morphed, size, 2)

def process(image_original):
    global toleranceFactor, softnessFactor, defringeFactor

    image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    mask = buildToleranceMask(image_hsv, toleranceFactor, 60)
    mask = blurMask(mask, softnessFactor)

    image_hsv[:,:,2] = np.bitwise_and(image_hsv[:,:,2], mask)

    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

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

while(cap.isOpened()):
    # Capture frame-by-frame
    if play or readOne:
        ret, frame = cap.read()
        if ret == True:
            resultFrame = frame
        readOne = False

    key = cv2.waitKey(25)
    if key & 0xFF == 27:
        break
    
    if key & 0xFF == 32:
        play = not play

    # Display the resulting frame
    cv2.imshow(windowName, process(resultFrame))

    # display progress trackbar
    cFrameNo = cap.get(cv2.CAP_PROP_POS_FRAMES); 
    cv2.createTrackbar("Progress", windowName, int(100 * cFrameNo / capLength) , 100, onPositionChange)

    # other trackbars
    cv2.createTrackbar("Tolerance", windowName, toleranceFactor, 100, lambda *args: onChangeTolerance())
    cv2.createTrackbar("Softness", windowName, softnessFactor, 100, lambda *args: onChangeSoftness())
    cv2.createTrackbar("Defringe", windowName, defringeFactor, 100, lambda *args: onChangeDefringe())

cv2.destroyAllWindows()