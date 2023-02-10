import pylab as plt
import cv2, os
from pylab import subplot, imshow, show, close
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Correct for skew
    image = deskew(image, max_angle=10)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply Otsu thresholding to enhance the text
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the image
    thresh = cv2.bitwise_not(thresh)

    return thresh

def deskew(image, max_angle):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the skew angle
    angle = cv2.minAreaRect(cv2.findNonZero(gray))[-1]

    # Check if the image is already deskewed
    if angle == 0 or abs(angle)<= max_angle:
        return image

    # Otherwise, deskew the image
    else:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

def straighten_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blur, 50, 150)

    # Find the contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    contour = max(contours, key=cv2.contourArea)

    # Get the rectangle bounding the contour
    rect = cv2.minAreaRect(contour)

    # Get the rotation angle
    angle = rect[2]

    # Get the center of the rectangle
    (cx, cy) = rect[0]

    # Get the width and height of the rectangle
    (w, h) = rect[1]

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Rotate the image
    straightened = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Return the straightened image
    return straightened

img_path = os.path.join('scanned', '20221210_092532.jpg')

raw = cv2.imread(img_path)
img = process_image(img_path)
deskewed = straighten_image(img_path)

subplot(221)
imshow(raw)
subplot(222)
imshow(img)
subplot(224)
imshow(deskewed)
show()
close()
