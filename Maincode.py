import cv2 
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import griddata
import scipy
from google.colab.patches import cv2_imshow


#Reading the image
I=cv2.imread('a.jpg');
if(I.ndim==3):
    I= cv2.cvtColor(I, cv2.COLOR_RGB2GRAY) # Grayscale conversion of image
plt.imshow(I,cmap = 'gray');
print(I.shape);


#Rescaling the image
scale_factor = 0.5;
# scale_factor = 0.2;
W = int(I.shape[1]*scale_factor);
H = int(I.shape[0]*scale_factor);
dimensions = (W,H);
print(dimensions);
re_I = cv2.resize(I,dimensions,interpolation = cv2.INTER_AREA);

plt.imshow(re_I,cmap = 'gray');



#Image Binarization using adaptive thresholding

# I_blur = cv2.medianBlur(re_I,5); # Helps in the smoothning out of the background lines
# # I_blur = cv2.medianBlur(re_I,5); # Helps in the smoothning out of the background lines
# I_thres = cv2.adaptiveThreshold(I_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,5);
# """
# I_thres = np.zeros((I_blur.shape));
# for i in range(0,I_blur.shape[0]):
#   for j in range(0,I_blur.shape[1]):
#     if (I_blur[i,j] >= 90):
#       I_thres[i,j] = 255;
#     else:
#       I_thres[i,j] = 0;
# """
# plt.imshow(I_thres,cmap = 'gray');


I_blur = cv2.medianBlur(re_I,7); # Helps in the smoothning out of the background lines


otsu_threshold, I_thres  = cv2.threshold(
I_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
print("Obtained threshold: ", otsu_threshold)

plt.imshow(I_thres,cmap = 'gray');


# I_thres = cv2.adaptiveThreshold(I_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,5);
# """
# I_thres = np.zeros((I_blur.shape));
# for i in range(0,I_blur.shape[0]):
#   for j in range(0,I_blur.shape[1]):
#     if (I_blur[i,j] >= otsu_threshold):
#       I_thres[i,j] = 255;
#     else:
#       I_thres[i,j] = 0;
# # """
# plt.imshow(I_thres,cmap = 'gray');





kernel = np.ones((7,7),np.uint8);

# kernel1 = np.ones((3,3),np.uint8);

# Firstly we erode the image, which increases the thickness of the black line
I_eroded = cv2.erode(I_thres,kernel,iterations = 1);
plt.imshow(I_eroded,cmap = 'gray');


I_dilated = cv2.dilate(I_eroded,kernel,iterations = 1);
plt.imshow(I_dilated,cmap = 'gray');





I_dilated = 255 - I_dilated;
cv2_imshow(I_dilated);








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
cv2_imshow(I_dilated)

# # De-allocate any associated memory usage
# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyAllWindows()
 


# corners = cv.goodFeaturesToTrack(operatedImage,25,0.001,10)
# corners = np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv.circle(operatedImage,(x,y),10,200,0)
# plt.imshow(operatedImage),plt.show()













I_copy = I_dilated.copy();
I_copy = cv2.medianBlur(I_copy,5);


contours,hierarchy = cv2.findContours(I_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);


print("Number of Contours found = " + str(len(contours)))

cv2_imshow(I_copy);
plt.show();

image = np.zeros((I_copy.shape));
print((I_copy.shape));
#We change the image into a white background by setting all intensity values as 255
for i in range(0,I_copy.shape[0]):
  for j in range(0,I_copy.shape[1]):
    image[i,j] = 255;



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
    print(ratio);
    #CASE: When we have a line segment
    if(ratio > 3):
      #Finding the endpoints of the line after checking if it is horizontal or vertical
      #Case 1: When the line is vertical inclined
      if(rotation_angle < 45):
        top = tuple(contour[contour[:,:,1].argmin()][0])
        bottom = tuple(contour[contour[:,:,1].argmax()][0])
        cv2.line(image,top,bottom, [0,0,0], 3); 
        Line.append((top,bottom));
      else:
        left = tuple(contour[contour[:,:,0].argmin()][0])
        right = tuple(contour[contour[:,:,0].argmax()][0])
        cv2.line(image,left,right, [0,0,0], 3);
        Line.append((left,right));

    elif(ratio > 0 and ratio <=3):
      cv2.ellipse(image,ellipse,[0,0,0],3);

  if (len(contour) < 5 and len(contour) >=2):
    if(rotation_angle < 45):
      top = tuple(contour[contour[:,:,1].argmin()][0])
      bottom = tuple(contour[contour[:,:,1].argmax()][0])
      cv2.line(image,top,bottom, [0,0,0], 3); 
      Line.append((top,bottom));
    else:
      left = tuple(contour[contour[:,:,0].argmin()][0])
      right = tuple(contour[contour[:,:,0].argmax()][0])
      cv2.line(image,left,right, [0,0,0], 3);
      Line.append((left,right));


cv2_imshow(image);











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
        cv2.line(image,a,(x_m,y_m), [0,0,0], 3);
        cv2.line(image,b,(x_m,y_m), [0,0,0], 3);

  


cv2_imshow(image);














