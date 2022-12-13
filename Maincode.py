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
st.image(img,width = 200)

#Converting the image to grayscale
I=cv2.imread(choice);
if(I.ndim==3):
    I= cv2.cvtColor(I, cv2.COLOR_RGB2GRAY) # Grayscale conversion of image
st.image(I,width=200);

#Rescaling the image
scale_factor = 0.5;
# scale_factor = 0.2;
W = int(I.shape[1]*scale_factor);
H = int(I.shape[0]*scale_factor);
dimensions = (W,H);
print(dimensions);
re_I = cv2.resize(I,dimensions,interpolation = cv2.INTER_AREA);

st.image(re_I,width=200);

# Helps in the smoothning out of the background lines

slider1 = st.select_slider("Blur Kernel Size",options=["3","5","7","9","11","13","15"])
st.write(slider1)
I_blur = cv2.medianBlur(re_I,int(slider1))
         
otsu_threshold, I_thres  = cv2.threshold(
I_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
st.write("Obtained threshold: ", otsu_threshold)

st.image(I_thres,width =200)

#Performing Erosion and Dilation
kernel = np.ones((7,7),np.uint8);

# Firstly we erode the image, which increases the thickness of the black line
I_eroded = cv2.erode(I_thres,kernel,iterations = 1);
plt.imshow(I_eroded,cmap = 'gray');
st.image(I_eroded,width=200);

I_dilated = cv2.dilate(I_eroded,kernel,iterations = 1);
st.image(I_dilated,width = 200);

#Conversion to Black and white
I_dilated = 255 - I_dilated;
st.image(I_dilated,width=200);


operatedImage = np.float32(I_dilated)

# apply the cv2.cornerHarris method
# to detect the corners with appropriate
# values as input parameters

dest = cv2.cornerHarris(operatedImage, 10,25, 0.07)
# Results are marked through the dilated corners
dest = cv2.dilate(dest, None)
print(I_dilated.shape);
# Reverting back to the original image,
# with optimal threshold value
I_dilated[dest > 0.01 * dest.max()]= 0;

# the window showing output image with corners
st.image(I_dilated,width = 200);


###########################################################
I_copy = I_dilated.copy();
I_copy = cv2.medianBlur(I_copy,5);


contours,hierarchy = cv2.findContours(I_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);


st.write("Number of Contours found = ",str(len(contours)))

st.image(I_copy)

image1 = np.zeros((I_copy.shape));
#We change the image into a white background by setting all intensity values as 255
for i in range(0,I_copy.shape[0]):
  for j in range(0,I_copy.shape[1]):
    image1[i,j] = 255;


st.image(image1)
#Storing all the line endpoints in a tuple
Line = [];
i = 1;
#Fitting an ellipse on the contours which form the image and disregarding the unrequired ones
for i in range(0,(len(contours))):

  contour = contours[i]
  
  if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour);
    i = i+1;
    print(i);

    (x_centre,y_centre),(minor_axis_diameter,major_axis_diameter),rotation_angle = ellipse;
    if(minor_axis_diameter != 0):
      ratio = major_axis_diameter/minor_axis_diameter;
    else:
      ratio = 10;
    
    #CASE: When we have a line segment
    if(ratio > 3):
      #Finding the endpoints of the line after checking if it is horizontal or vertical
      #Case 1: When the line is vertical inclined
      if(rotation_angle < 45):
        top = tuple(contour[contour[:,:,1].argmin()][0])
        bottom = tuple(contour[contour[:,:,1].argmax()][0])
        cv2.line(image1,top,bottom, [0,0,0], 3); 
        Line.append((top,bottom));
      else:
        left = tuple(contour[contour[:,:,0].argmin()][0])
        right = tuple(contour[contour[:,:,0].argmax()][0])
        cv2.line(image1,left,right, [0,0,0], 3);
        Line.append((left,right));

    elif(ratio > 0 and ratio <=3):
      cv2.ellipse(image1,ellipse,[0,0,0],3);

  if (len(contour) < 5 and len(contour) >=2):
    if(rotation_angle < 45):
      top = tuple(contour[contour[:,:,1].argmin()][0])
      bottom = tuple(contour[contour[:,:,1].argmax()][0])
      cv2.line(image1,top,bottom, [0,0,0], 3); 
      Line.append((top,bottom));
    else:
      left = tuple(contour[contour[:,:,0].argmin()][0])
      right = tuple(contour[contour[:,:,0].argmax()][0])
      cv2.line(image1,left,right, [0,0,0], 3);
      Line.append((left,right));


st.image(image1)













