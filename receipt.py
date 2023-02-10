import cv2, os
import pytesseract
import numpy as np
from fast_deskew import deskew_image

def readContent(img, conf):
    text = pytesseract.image_to_string(img, config=conf)
    level, passed = eval(text)
    return text, level, passed