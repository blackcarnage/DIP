import streamlit as st
from PIL import Image

import cv2 
import matplotlib.pyplot as plt
import os
import numpy as np
#from scipy.interpolate import griddata
import scipy
#from google.colab.patches import cv2_imshow



#App headings
st.title("Digital Image Processing")
st.header("End Semester Project")

# Making the user choose the image to convert
choice = st.selectbox("Images:",["a.jpg"])
img = Image.open(choice)
st.image(img)

#Converting the image to grayscale
I=cv2.imread(choice);
if(I.ndim==3):
    I= cv2.cvtColor(I, cv2.COLOR_RGB2GRAY) # Grayscale conversion of image
st.image(I,width="200");

#Rescaling the image
scale_factor = 0.5;
# scale_factor = 0.2;
W = int(I.shape[1]*scale_factor);
H = int(I.shape[0]*scale_factor);
dimensions = (W,H);
print(dimensions);
re_I = cv2.resize(I,dimensions,interpolation = cv2.INTER_AREA);

st.image(re_I,width="200");

# Helps in the smoothning out of the background lines

slider1 = st.select_slider("Blur Kernel Size",options=["3","5","7","9","11","13","15"])
st.write(slider1)
I_blur = cv2.medianBlur(re_I,int(slider1))
         
otsu_threshold, I_thres  = cv2.threshold(
I_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
st.write("Obtained threshold: ", otsu_threshold)

st.image(I_thres,width ="200")

#Performing Erosion and Dilation
kernel = np.ones((7,7),np.uint8);

# Firstly we erode the image, which increases the thickness of the black line
I_eroded = cv2.erode(I_thres,kernel,iterations = 1);
plt.imshow(I_eroded,cmap = 'gray');
st.image(I_eroded,width="200");

I_dilated = cv2.dilate(I_eroded,kernel,iterations = 1);
st.image(I_dilated,width = "200");

#Conversion to Black and white
I_dilated = 255 - I_dilated;
st.image(I_dilated,width="200");

















