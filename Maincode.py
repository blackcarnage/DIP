import streamlit as st
from PIL import Image
import pandas as pd
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

st.sidebar.write("""#### Choose your Parameters""")
slider0 = st.sidebar.select_slider("Image Resize Factor",options=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
slider1 = st.sidebar.select_slider("Blur Kernel Size",options=["3","5","7","9","11","13","15"])




# Making the user choose the image to convert
choice = st.sidebar.selectbox("Images:",["a.jpg","b.jpg","c.jpg","d.jpg"])
img = Image.open(choice)
st.image(img,width = 200)

#Converting the image to grayscale
I=cv2.imread(choice);
if(I.ndim==3):
    I= cv2.cvtColor(I, cv2.COLOR_RGB2GRAY) # Grayscale conversion of image
st.image(I,width=200);

#Rescaling the image

# slider0 = st.select_slider("Image Resize Factor",options=["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"])
# st.write(slider0)


scale_factor = float(slider0);
W = int(I.shape[1]*scale_factor);
H = int(I.shape[0]*scale_factor);
dimensions = (W,H);
print(dimensions);
re_I = cv2.resize(I,dimensions,interpolation = cv2.INTER_AREA);

st.image(re_I,width=200);

# Helps in the smoothning out of the background lines


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


dest = cv2.cornerHarris(operatedImage, 20,25,0.07)
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
    image1[i,j] = 1;



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

  if (len(contour) < 5 and len(contour) >5):
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


# st.image(image1)

#############################################################################
from itertools import combinations 
from numpy import inf
#finding all possible combinations of endpoints
val = combinations(Line, 2) ;
x = np.zeros(4);
y = np.zeros(4);
for index in val:
    
    #Line 1
  
    x[0] = index[1][0][0];
    y[0] = index[1][0][1];

    x[1] = index[1][1][0];
    y[1] = index[1][1][1];

    #Line 2  
    x[2] = index[0][0][0];
    y[2] = index[0][0][1];

    x[3] = index[0][1][0];
    y[3] = index[0][1][1];

    d1 = ((x[1] - x[2])**2 + (y[1] - y[2])**2)**0.5;
    d0 = ((x[0] - x[2])**2 + (y[0] - y[2])**2)**0.5;

    d3 = ((x[1] - x[3])**2 + (y[1] - y[3])**2)**0.5;
    d2 = ((x[0] - x[3])**2 + (y[0] - y[3])**2)**0.5;
    
    # when lines intersect
    if( d1<40 or d2< 40 or d3 <40 or d0 <40):
      A1 = y[3] - y[2];
      B1 = x[2] - x[3];
      C1 = A1*x[2] + B1*y[2];

      A2 = y[1] - y[0];
      B2 = x[0] - x[1];
      C2 = A2*x[0] + B2*y[0];
      
      Dx =  B2*C1 - B1*C2;
      Dy = A1*C2 - A2*C1;

      D = A1*B2 - A2*B1;
      
      x_m = (Dx/D);
      y_m = (Dy/D);
      if(x_m != inf and y_m != inf and x_m>0 and y_m>0):
        x_m = int(Dx/D);
        y_m = int(Dy/D);

        if(d0<40):
          a = (int(x[0]),int(y[0]));
          b = (int(x[2]),int(y[2]));
      
        elif(d1<40):
          a = (int(x[1]),int(y[1]));
          b = (int(x[2]),int(y[2]));
        elif(d2<40):
          a = (int(x[0]),int(y[0]));
          b = (int(x[3]),int(y[3]));
        else:
          a = (int(x[1]),int(y[1]));
          b = (int(x[3]),int(y[3]));
      
      
        print(x_m,y_m);
        cv2.line(image1,a,(x_m,y_m), [0,0,0], 3);
        cv2.line(image1,b,(x_m,y_m), [0,0,0], 3);

  


st.image(image1)













