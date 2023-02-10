import cv2, os
import numpy as np
from pylab import subplot, imshow, show, close

def canny_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    kern = np.ones((3,3), np.uint8)
    dil = cv2.dilate(thresh, kern, iterations=1)
    cann = cv2.Canny(dil, 50, 255)
    contour, hier = cv2.findContours(cann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contour, -1, (255,0,0), 2)
    for cont in contour:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,0,255), 2)
    
    subplot(221), imshow(img)
    subplot(222), imshow(gray, 'gray')
    subplot(223), imshow(dil, 'gray')
    subplot(224), imshow(cann, 'gray')
    show()
    close()
    
      
image = cv2.imread(os.path.join('scanned','20221210_092806.jpg'))
canny_process(image)