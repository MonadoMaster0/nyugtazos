import cv2
import numpy as np

def skewAngle(image, kernel=np.ones((3,3), np.uint8), r_factor = 1):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    img = cv2.dilate(img, kernel, iterations=1)

    cont, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cont, key=cv2.contourArea, reverse=True)
    # for c in cont:
    min = cv2.minAreaRect(c[r_factor])
    angle = min[-1]
    if angle < -45:
        angle = 90 + angle
    return angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(img, kernel=np.ones((2,10), np.uint8), r_factor=1):
    angle = skewAngle(img, kernel, r_factor)
    return rotateImage(img, angle)




