import cv2, os, random, pandas
import numpy as np
import pylab as plt

fileList = os.listdir('scanned')

data = pandas.DataFrame()

# plottoláshoz listák
processed = []
graphs = []
raw = []

for ind, file in enumerate(fileList):
    pair = {}
    img = cv2.imread(os.path.join('scanned', file))
    h, w = img.shape[:2] # Felveszem a kép dimenzióit. height, width -- a numpy először a magasságot adja utána a szélességet, a cv2 fordítva. A .shape method három értéket ad: (magasság, szélesség, csatornák száma)
    s = int(w*0.62) # En nem tövényszerű, de általánosságban az értékes információ a képszélesség 62 %-ánál kedődik magasságban
    img = img[s:h, 0:w] # Ezt a vágást itt ejtem meg. s-től h-ig vágom a magasságot, 0-tól w-ig a szélességet (numpy szintaxis, azaz magasság elől, szélesség utána) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Szürkekép
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] # Bináris kép készítése
    
    kern = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kern, iterations=1) # 3 pixeles bővítése
    
    cont, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Kontúrkeresés
    processed.append(thresh) # képek listához adása a későbbi megjelenítést könnyebbíti meg
    raw.append(img)
    
    # Listák a plottoláshoz
    x_plot = []
    y_plot = []
    for c in cont:
        x,y,w,h = cv2.boundingRect(c)
        if h*w*100/(img.shape[0]*img.shape[1]) < 0.3: # A bounding box területét a képméretéhez viszonyatva százalékosan levágja a kiugró értékeket
            x_plot.append(y)
            y_plot.append(w*h)
            color = (0,0,255)
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        if h*w > 10000:
            print(f'{h*w*100/(img.shape[0]*img.shape[1])}\t{w*h}\t{x}\t{y}') # Fejleszői cucc, a területszűrésnél a kép amit használtam, ott 10000 területű boxokat generált, amikeben már nem volt értékes tartalom
    graphs.append([x_plot,y_plot])


r = random.randint(0,len(processed)-1) # kiválaszt egy random képet amit kiplottol


print(data)

plt.subplot(221)
plt.plot(graphs[r][0],graphs[r][1])
plt.subplot(222)
plt.imshow(processed[r], 'gray')
plt.subplot(223)
plt.imshow(raw[r])
plt.show()
print(raw[r].shape)
        