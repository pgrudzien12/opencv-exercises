# Enter your code here
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def calcVariance(patch):
    x = cv2.Scharr(patch, -1, 1, 0)
    y = cv2.Scharr(patch, -1, 0, 1)

    return np.abs(x) + np.abs(y)


def displayZoomed(fragment, wndName, wndTitle=None):
    size = (400, 400)
    resizedFragment = cv2.resize(fragment, size)
    if wndTitle != None:
        cv2.setWindowTitle(wndName, wndTitle)
    cv2.imshow(wndName, resizedFragment)

def getPatch(xy, gray, img)
    colorPatch = img[xy[1] - 15:xy[1] + 15, xy[0] - 15:xy[0] + 15]
    grayPatch = gray[xy[1] - 15:xy[1] + 15, xy[0] - 15:xy[0] + 15]
    return (grayPatch, colorPatch)

def pickBestAround(xy, values, image):
    bestV = 0
    best_xy = None
    for move in np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]):
        move_s = move * 30
        xy_m = xy + move_s
        if xy_m[xy_m<0].sum() < 0:
            continue
        patch, _ = values[xy_m[1] - 15:xy_m[1] + 15, xy_m[0] - 15:xy_m[0] + 15]
        variance = 1/calcVariance(patch).sum()
        if variance > bestV:
            bestV = variance
            best_xy = xy_m

        return getPatch(best_xy)

def onMouse(action, x, y, flags, img):
    if action == cv2.EVENT_LBUTTONDOWN or action == cv2.EVENT_RBUTTONDOWN:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = img_hsv[:, :, 0]
        xy_m = np.array((x,y)) + np.array((-15, -15))
        if xy_m[xy_m<0].sum() < 0:
            return
        patch = gray[y-15:y+15, x-15:x+15]
        cpatch = img[y-15:y+15, x-15:x+15]
        variance = calcVariance(patch)
        displayZoomed(cpatch, "Variance",
                      "Patch {0}".format(np.sum(variance)))

        (grayPatch, colorPatch) = pickBestAround((x, y), gray, img)
        variance = calcVariance(grayPatch)
        displayZoomed(colorPatch, "Best Patch",
                      "Best Patch V={0}".format(np.sum(variance)))
        if action == cv2.EVENT_LBUTTONDOWN:
            return
        src_mask = np.ones_like(patch) * 255
        cv2.seamlessClone(
            colorPatch, img, src_mask, (x, y), cv2.NORMAL_CLONE, blend=img)

img = cv2.imread("blemish-removal/blemish.png", 1)
# Make a dummy image, will be useful to clear the drawing
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", onMouse, img)

k = 0
while k != 27:
    cv2.imshow("Window", img)
    k = cv2.waitKey(20)

cv2.destroyAllWindows()
