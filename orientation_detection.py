import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('1_grasp_42_0_colour.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh, 1, 2)

contour = contours[0]
for i in range(1,len(contours)):
    if contours[i].shape[0]>4:
      cnt = contours[i]
      contour = np.concatenate([contour, cnt], axis=0)


ellipse = cv2.fitEllipse(contour)
cv2.ellipse(img,ellipse,(0,255,0),2)
plt.imshow(img,'gray')
plt.show()

##################################
img = cv2.imread('1_grasp_42_0_colour.jpg',0)
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
plt.imshow(img,'gray')
plt.show()
