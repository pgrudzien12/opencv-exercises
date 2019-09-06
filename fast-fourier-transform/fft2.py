# Enter your code here
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def magSpectrum(patch):
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    return magnitude_spectrum

def displayZoomed(wndName, fragment, wndTitle = None):
    size = (400,400)
    resizedFragment = cv2.resize(fragment, size)
    if wndTitle != None:
            cv2.setWindowTitle(wndName, wndTitle)
    cv2.imshow(wndName, resizedFragment)

def onMouse(action, x, y, flags, userdata): 
    if action==cv2.EVENT_LBUTTONDOWN:
        # Mark the center
        img = userdata[y-15:y+15,x-15:x+15]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        magnitude_spectrum = magSpectrum(img)
        
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'coolwarm')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()


source = cv2.imread("blemish-removal/blemish.png",1)
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()
cv2.namedWindow("Window")
cv2.namedWindow("patch")
cv2.namedWindow("Variance")
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
