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
st.image(I);

#Rescaling the image
scale_factor = 0.5;
# scale_factor = 0.2;
W = int(I.shape[1]*scale_factor);
H = int(I.shape[0]*scale_factor);
dimensions = (W,H);
print(dimensions);
re_I = cv2.resize(I,dimensions,interpolation = cv2.INTER_AREA);

st.image(re_I);


level = st.slider("Select the level", 1, 15,(1,15),2)
slider1 = st.slider("Blur Kernel Size",
I_blur = cv2.medianBlur(re_I,slider1); # Helps in the smoothning out of the background lines


otsu_threshold, I_thres  = cv2.threshold(
I_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
st.write("Obtained threshold: ", otsu_threshold)

st.image(I_blur)



















