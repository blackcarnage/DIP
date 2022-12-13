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


















