import pandas as pd
import pylab as plt
import numpy as np
import cv2, random, os


rand = random.choice(os.listdir('scanned'))
PATH = f'scanned\\{rand}'
TEMP1 = cv2.imread('templates\\START.jpg', cv2.IMREAD_GRAYSCALE)
TEMP2 = cv2.imread('templates\\END.jpg', cv2.IMREAD_GRAYSCALE)
info_depth = 0

def analyze(path: str, depth_ratio=0.62,  dil_size: tuple=(3,3), dil_iter=1, area_cutsize=1, cutmode = 'percent'):
    """Creates a pre-anaylzation of image. Determines valuable information range on the y axis.
    
    ...
    
    Parameters
    ------------
    path: str
        Path to the image file which to be analyzed.
        
    depth_ratio: float
        A ratio between the width of the image and the depth where valuable information starts.
        Recommended value is 0.62 (should not ecceed 0.7)
        
    dil_size: tuple
        A tuple of two integers, the kernel size for dilation.
        The greater the kernel size the larger boxes the boxes will be.
        Oversized kernel leads to dataloss, undersized kernel causes noise.
        
    dil_iter: int
        Number of iteration for dilation convolution.
        Can be an option if kernel size incrementation does not lead to success.
        
    area_cutsize: float
        A threshold value for proportional box area to image area.
        Used to cut out overly large boxes like edge area of contaminations.
        Default value is 1 which means that practicly it is not used.
    
    Returns
    -------
    x_plot: list
        List of depth values in pixels of all the boxes.
    y_plot: list
        List of areas in pixels of all the boxes.
    image: ndarray
        The image array with bounding boxes
    """
    
    img = cv2.imread(path)
    h, w = img.shape[:2] # Felveszem a kép dimenzióit. height, width -- a numpy először a magasságot adja utána a szélességet, a cv2 fordítva. A .shape method három értéket ad: (magasság, szélesség, csatornák száma)
    loc, loc2, _ = search(path)
    s = int(w*depth_ratio) if cutmode == 'percent' else loc[1] # En nem tövényszerű, de általánosságban az értékes információ a képszélesség 62 %-ánál kedődik magasságban
    q = h if cutmode == 'percent' else loc2[1]
    img = img[s:q, 0:w] # Ezt a vágást itt ejtem meg. s-től h-ig vágom a magasságot, 0-tól w-ig a szélességet (numpy szintaxis, azaz magasság elől, szélesség utána) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Szürkekép
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] # Bináris kép készítése
    
    kern = np.ones(dil_size, np.uint8)
    thresh = cv2.dilate(thresh, kern, iterations=dil_iter) # 3 pixeles bővítése
    
    cont, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Kontúrkeresés
    
    x_plot = []
    y_plot = []
    for c in cont:
        x,y,w,h = cv2.boundingRect(c)
        if h*w*100/(img.shape[0]*img.shape[1]) < area_cutsize: # A bounding box területét a képméretéhez viszonyatva százalékosan levágja a kiugró értékeket
            x_plot.append(y)
            y_plot.append(w*h)
            color = (255,0,0)
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 1)
    
    return x_plot, y_plot, img

def deriv(tup, level):
    if level > 0:
        x, y = tup[0], tup[1]
        i0 = 0
        j0 = 0
        der = []
        for i, j in zip(x,y):
            d = (j-j0)/(i-i0+0.000000000001)
            der.append(d)
            j0 = j
            i0 = i
        return deriv((tup[0], der), level-1)
    elif level <= 0:
        return tup[0], tup[1]
            
def smooth(tup, size= 5, kernel=None):
    """Smooths out function via moving average

    Args:
        tup (tuple): x and y values
        size (int, optional): _description_. Defaults to 5.
        kernel (1D array or list, optional): convolution kernel. Defaults to None.

    Returns:
        x and y lists
    """
    func = list(zip(tup[0], tup[1]))
    kern = [1 for _ in range(size)] if kernel == None else kernel
    smoothed = []
    for i in range(len(func)):
        if i >= size:
            s = [func[i-z][1]*kern[z] for z in range(size)]
        elif i < size:
            s = [func[i-z][1]*kern[z] for z in range(size-i)]
        p = np.average(s)
        smoothed.append(p)
    return tup[0], smoothed

def search(path: str):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, TEMP1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    result2 = cv2.matchTemplate(gray, TEMP2, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)
    cv2.circle(img, max_loc, 10, (255,0,0), 10)
    cv2.circle(img, max_loc2, 10, (0,255,0), 10)
    return max_loc, max_loc2, img

try:
    x, y, image = analyze(PATH, cutmode='temp')
except cv2.error:
    x, y, image = search(PATH)

plt.subplot(121), plt.plot(x, y)
plt.subplot(122), plt.imshow(image)
plt.show()