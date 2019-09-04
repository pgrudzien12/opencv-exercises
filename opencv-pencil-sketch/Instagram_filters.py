
# coding: utf-8

# # <font style="color:rgb(50,120,229)">Introduction</font>

# Instagram is a photo-enhancing and sharing app for mobile phones. The name came from “Instant Camera and TeleGram”. It allows users to apply different filters to their pictures and share it. It was launched in September 2010 and was an instant hit among its users with 1 Million registered users within 2 months and 10 million in a year. It was acquired by Facebook in 2012 for $1 Billion. [**You know what is cool? A billion dollars!**](https://www.youtube.com/watch?v=4e0n7vTLz1U)
# 
# There are many photographic filters present in the app like Clarendon, Gingham, Lark, Moon, X-Pro II, Nashville, Sutro, Kelvin and many more. Most filters can be designed using the method described in color enhancement using curves section. 
# 
# In this part of the project, You will design 2 instagram-like filter on your own. 
# 
# 1. Pencil Sketch Filter - This will generate a sketch of the given image as shown in the output below.
# 1. Cartoon Filter using - This should produce a cartoonified output of the input image.
# 
# Given below is the expected output for the two filters. Looking at the output, you should be able to figure out what processing can produce those results.
# 
# You can also come up with your own filter and show them in the discussion forum. 

# | <center> <a href="https://www.dropbox.com/s/0w8j6ejacfw3lwt/trump.jpg?dl=1"><img src = "https://www.dropbox.com/s/0w8j6ejacfw3lwt/trump.jpg?dl=1" width=500/></a></center> | <center> <a href="https://www.dropbox.com/s/yegx6u3bx8otvh5/pencilSketch.jpg?dl=1"><img src = "https://www.dropbox.com/s/yegx6u3bx8otvh5/pencilSketch.jpg?dl=1" width=500/></a></center> | <center> <a href="https://www.dropbox.com/s/rttwy9wszo76jse/cartoon.jpg?dl=1"><img src = "https://www.dropbox.com/s/rttwy9wszo76jse/cartoon.jpg?dl=1" width=500/></a></center> | 
# | -------- | -------- | -------- | 
# | <center>Original Image</center>     | <center>Pencil Sketch Result</center>     | <center>Cartoon Filter Result</center>     |
# 

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'


# In[33]:


def cartoonify(image, arguments=0):
    ### YOUR CODE HERE
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(img_gray, 3)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(image, 30, 80, 80)
    contours = pencilSketch(image)
    return cv2.bitwise_and(color, color, mask = contours[:,:,0])


# In[34]:


def pencilSketch(image, arguments=0):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (5,5), 1.5)
    lapl = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3, scale=2, delta = 1)
    _, dst = cv2.threshold(lapl, 16, 255, cv2.THRESH_BINARY)
                    
    return cv2.cvtColor(255-dst, cv2.COLOR_GRAY2BGR)


# In[35]:


imagePath = "data/images/trump.jpg"
image = cv2.imread(imagePath)

cartoonImage = cartoonify(image)
pencilSketchImage = pencilSketch(image)


# In[36]:


targetImg = cv2.imread("data/images/cartoon.jpg")
cv2.imshow('targetImg',targetImg)

plt.figure(figsize=[20,10])
plt.subplot(131);plt.imshow(image[:,:,::-1]);
plt.subplot(132);plt.imshow(cartoonImage[:,:,::-1]);
plt.subplot(133);plt.imshow(pencilSketchImage[:,:,::-1]);
plt.show()


# In[ ]:




