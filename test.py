import cv2, os, math
import pylab as plt
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CONFIG_1 = r'--psm 6 --oem 3 -l hun'
CONFIG_2 = r'--psm 1 --oem 3 -l hun'

image = cv2.imread(os.path.join('scanned', '20221127_192505.jpg'))
blank = np.ones(image.shape[:2], np.uint8)*255
ho = np.copy(image)
blank = cv2.merge((blank, blank, blank))
print(image.shape)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)
kern = np.ones((5,7), np.uint8)
thresh =cv2.dilate(thresh, kern, iterations=1)

cont, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2. CHAIN_APPROX_SIMPLE)
cv2.drawContours(blank, cont, -1, (0,0,0), 1)
x_plot = []
y_plot = []
for c in cont:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(ho, (x,y),(x+w,y+h), (255,0,0), 1)
    x_plot.append(y)
    y_plot.append(w*h)


# plt.plot(x_plot, y_plot)
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(thresh, 'gray')
plt.subplot(223)
plt.imshow(blank)
plt.subplot(224)
plt.imshow(ho)
plt.show()