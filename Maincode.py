import streamlit as st
from PIL import Image

import cv2 
import matplotlib.pyplot as plt
#import os
import numpy as np
#from scipy.interpolate import griddata
import scipy
#from google.colab.patches import cv2_imshow




st.title("Digital Image Processing")
st.header("End Semester Project")

img = Image.open("a.jpg");
st.image(img);














