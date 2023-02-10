import numpy as np
import cv2, os
import pytesseract
import pylab as plt
from deskew import deskew

def textEval(text: str, exclusionLimit = 5):
    """Returns a percentage value of utility charachters in text and a boolean value based on exclusion limit.

    :param str text: The text that will be processed.
    :param int exclusionLimit: This number indicates the limit, how many percent of the text must unusual charachters to be evaluated as FAIL. 
    """
    exclusion_chars = [':', '.', '%', ',', '/', '"', '\'', '-', ')', '(']
    wrong_level = 0
    for ex_ch in exclusion_chars:
        wrong_level += text.count(ex_ch)
    percentage = round(wrong_level/len(text)*100, 2)
    passed = True if percentage < exclusionLimit else False
    return percentage, passed

def readFiles(folderPath: str):
    """
    Creates a dictionary from the files.

    :param str folderPath: Path to the folder where the images are. Only accepts .jpeg

    """
    log = []
    for file in os.listdir('scanned'):
        entry = {}
        entry['path'] = os.path.join('scanned', file)
        entry['date'] = f'{file[:4]}.{file[4:6]}.{file[6:8]}'
        entry['time'] = f'{file[9:11]}:{file[11:13]}:{file[13:15]}'
        entry['status'] = 'processing'
        entry['config'] = ''
        entry['better_result'] = None
        entry['excluded'] = 0
        log.append(entry)
    return log

def taskBar(progress, total):
    """
    Simple taskbar to show progress.
    """
    perc = progress/float(total)*100
    bar = '█' * int(perc) + '-' * (100-int(perc))
    print(f'|{bar}| {perc:.2f} %\r', end=f'\r')

def preProcess(image):
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)

    
    print(temp)
    return temp


image = cv2.imread('skewed.jpg')
imaged = deskew(image)


# print(image)
plt.subplots(1,2)
plt.imshow(image, 'gray')
plt.show()

