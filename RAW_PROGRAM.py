import cv2, os
from text_eval import eval
import pytesseract
import numpy as np
from deskew import deskew

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CONFIG_1 = r'--psm 6 --oem 3 -l hun'
CONFIG_2 = r'--psm 1 --oem 3 -l hun'
logfile = open('log.txt', 'w')
logfile.write(f"Name\tPath\tDate\tTime\tStatus\tConfig\tExcluded percentage\n")
logfile.close()

def taskBar(progress, total):
    perc = progress/float(total)*100
    bar = '█' * int(perc) + '-' * (100-int(perc))
    print(f'|{bar}| {perc:.2f} %\r', end=f'\r')


def readFiles():
    log = []
    for file in os.listdir('scanned'):
        entry = {}
        entry['path'] = os.path.join('scanned', file)
        entry['date'] = f'{file[:4]}.{file[4:6]}.{file[6:8]}'
        entry['time'] = f'{file[9:11]}:{file[11:13]}:{file[13:15]}'
        entry['status'] = 'processing'
        log.append(entry)
    return log

def readContent(img, conf):
    text = pytesseract.image_to_string(img, config=conf)
    level, passed = eval(text)
    return text, level, passed

def noiseRemoval(image):
    kernel = np.ones((1,1), np.uint8)
    temp = cv2.dilate(image, kernel, iterations=1)
    temp = cv2.erode(temp, kernel, iterations=1)
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    temp = cv2.medianBlur(temp, 3)
    return temp

def textGen(image, entry):
    curr_text, curr_level, passed = readContent(image, CONFIG_1)
    if passed:
        entry['status'] = 'done'
        entry['configuration'] = 'CONFIG_1'
        entry['better_result'] = '-'
        entry['excluded'] = curr_level
        with open(os.path.join('text',f"{entry['path'][-19:-4]}.txt"), 'w', encoding='utf-8') as curr_file:
            curr_file.write(curr_text)
    else:
        curr_text1, curr_level1, passed1 = readContent(image, CONFIG_2)
        if passed1:
            entry['status'] = 'done'
            entry['better_result'] = 'CONFIG_1' if curr_level < curr_level1 else 'CONFIG_2'
            entry['configuration'] = 'CONFIG_2'
            entry['excluded'] = curr_level1
            with open(os.path.join('text',f"{entry['path'][-19:-4]}.txt"), 'w', encoding='utf-8') as curr_file:
                curr_file.write(curr_text1)
        else:
            entry['status'] = 'failed'
            entry['better_result'] = 'CONFIG_1' if curr_level < curr_level1 else 'CONFIG_2'
            entry['configuration'] = None
            entry['excluded'] = curr_level
            with open(os.path.join('text',f"{entry['path'][-19:-4]}.txt"), 'w', encoding='utf-8') as curr_file:
                curr_file.write(curr_text if curr_level < curr_level1 else curr_text1)
    

    with open('log.txt', 'a') as ll:
        ll.write(f"{entry['path'][-19:]}\t{entry['path']}\t{entry['date']}\t{entry['time']}\t{entry['status']}\t{entry['configuration']}\t{entry['excluded']}\t{entry['better_result']}\n")

def processImages(log):
    for index, entry in enumerate(log):
        taskBar(index, len(log))
        img = cv2.imread(entry['path'], cv2.IMREAD_UNCHANGED)
        img = deskew(img, r_factor=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        processed = noiseRemoval(img[1])
        textGen(processed, entry)
    print('')




log = readFiles()
raw = processImages(log)
