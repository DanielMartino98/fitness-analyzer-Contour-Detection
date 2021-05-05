import cv2
from scipy.signal import find_peaks
import os.path
import imutils
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
import numpy as np

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Neues Bild (Numpy-Array) erzeugen, das mit einer bestimmten Farbe in RGB gefüllt ist"""
    #schwarzes und leeres Bild erstellen
    image = np.zeros((height, width, 3), np.uint8)
    #da OpenCV BGR verwendet, müssen die Farben konvertiert werden
    color = tuple(reversed(rgb_color))
    #Mit gewünschter Farbe füllen
    image[:] = color
    return image

def takeClosest(myList, myNumber, start, end):
    difference = abs(myNumber-myList[0])
    for i in range(start, end, 1):
        if abs(myList[i] - myNumber) < difference:
            difference = abs(myList[i] - myNumber)
            searched = i
    return searched

path = r'D:\...\NAME.mp4'
DIR = r'C:\...\NAME' #Ordner erstellen falls noch nicht vorhanden

for fname in os.listdir(DIR):  #so lösche ich alle Dateien die mit der Bezeichnung 'frame' beginnen
    if fname.startswith("frame"):
        os.remove(os.path.join(DIR, fname))


#farbfiltern
lower_color = np.array([20, 60, 20], np.uint8) #gelb
higher_color = np.array([55, 255, 255], np.uint8) #ohne dunkelgrün
#Farbspektrum für blau
'''
#lower_color = np.array([80, 50, 50])  
#higher_color = np.array([130, 255, 255])
'''

#Falls die Bearbeitungszeit gestoppt werden soll, muss folgende Zeile eingeblendet werden
#t0 = time.time()

height_log = []
point_list = []
nummer = 0
half_window_rectangle = 50

cap = cv2.VideoCapture(path)

while(True): #Jeden Frame laden
    ret, im = cap.read()
    nummer += 1

    try:
        if im.shape[0] > 4000:
            im = cv2.resize(im, (int(im.shape[1] / 7), int(im.shape[0] / 7)))
        elif im.shape[0] > 2000:
            im = cv2.resize(im, (int(im.shape[1] / 5), int(im.shape[0] / 5)))
        elif im.shape[0] > 1000:
            im = cv2.resize(im, (int(im.shape[1] / 3), int(im.shape[0] / 3)))

        #Falls das Bild um 90° gedreht werden soll, muss die folgende Zeile eingeblendet werden
        #im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        frame = im.copy()
    except:
        break

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, lower_color, higher_color)

    if nummer > 1: #Für jedem Frame nach dem ersten Frame mit detektiertem Orientierungspunkt
        if height_log ==[] or point_list ==[]:
            case=0
            frame_threshed = frame_threshed
        elif height_log[-1] < half_window_rectangle and point_list[-1][0] < half_window_rectangle:
            case = 0
            frame_threshed = frame_threshed[0: height_log[-1] + half_window_rectangle, 0: point_list[-1][0] + half_window_rectangle]
            frame = frame[0: height_log[-1] + half_window_rectangle,
                             0: point_list[-1][0] + half_window_rectangle]
        elif height_log[-1] == 0:
            case = 0
            frame_threshed = frame_threshed
        elif height_log[-1] < half_window_rectangle:
            case = 3
            frame_threshed = frame_threshed[0: height_log[-1] + half_window_rectangle, point_list[-1][0] - half_window_rectangle: point_list[-1][0] + half_window_rectangle]
            frame = frame[0: height_log[-1] + half_window_rectangle,
                             point_list[-1][0] - half_window_rectangle: point_list[-1][0] + half_window_rectangle]
        elif point_list[-1][0] < half_window_rectangle:
            case = 2
            frame_threshed = frame_threshed[height_log[-1] - half_window_rectangle: height_log[-1] + half_window_rectangle, 0: point_list[-1][0] + half_window_rectangle]
            frame = frame[height_log[-1] - half_window_rectangle: height_log[-1] + half_window_rectangle,
                            0: point_list[-1][0] + half_window_rectangle]
        else:
            case = 1
            frame_threshed = frame_threshed[height_log[-1] - half_window_rectangle: height_log[-1]  + half_window_rectangle, point_list[-1][0] - half_window_rectangle: point_list[-1][0] + half_window_rectangle]
            frame = frame[height_log[-1] - half_window_rectangle: height_log[-1] + half_window_rectangle,
                             point_list[-1][0] - half_window_rectangle: point_list[-1][0] + half_window_rectangle]
    else:
        case = 0
    cnts = cv2.findContours(frame_threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finde Konturen
    cnts = imutils.grab_contours(cnts)
    center = None

    i = 0
    Circularity_list = []
    found_points_list = []
    contour_list = []
    for c in cnts:
    #Mittelpunkt der Konturen ermitteln
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        try:
            Circularity = (4*3.14*area)/(perimeter)**2
        except:
            Circularity = 0

        if area > 40 and Circularity > 0.5:
            i += 1
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Übersetze es wieder in Absolutkoordinaten
            if case == 1:
                cX = point_list[-1][0] - half_window_rectangle + cX
                cY = height_log[-1] - half_window_rectangle + cY
            elif case == 2:
                cY = height_log[-1] - half_window_rectangle + cY
            elif case == 3:
                cX = point_list[-1][0] - half_window_rectangle + cX

            Circularity_list.append(Circularity)
            found_points_list.append((cX, cY))
            contour_list.append(c)

        else:
            continue
    try:
        if len(Circularity_list) != 0:
            index = Circularity_list.index(max(Circularity_list))
            height_log.append(found_points_list[index][1])
            point_list.append(found_points_list[index])
            # Zeichne Kontur und Schwerpunkt auf das Bild
            cv2.drawContours(frame, [contour_list[index]], -1, (0, 0, 255), 2)
            cv2.circle(frame, point_list[-1], 1, (255, 0, 0), -1)
        else:
            height_log.append(height_log[-1])
            point_list.append(point_list[-1])
    except:
        height_log.append(0)


#Falls die Frames während der Bearbeitung angezeigt werden sollen müssen die folgenden Zeilen eingeblendet werden
'''
    cv2.imshow("threshed frame", frame_threshed)
    cv2.imshow("Original Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''

#Falls die Bearbeitungszeit gestoppt werden soll, müssen folgende Zeilen eingeblendet werden
#t1 = time.time() #Bis zu dieser Zeile wird die benötigte Bearbeitungszeit gestoppt
#print("Bearbeitungszeit: ", t1-t0, " s")

x = range(0, len(height_log))
y = height_log
y_smoothed = savgol_filter(y, 51, 3) #Kurve glätten

average = sum(y_smoothed)/len(y_smoothed) #Mittelwert der Liste ermitteln
maxima_log = find_peaks(y_smoothed, distance=50, height=(average, max(y_smoothed)))[0] #finde Maxima

#Falls die Kurve ausgegeben werden soll, müssen die folgenden Zeilen eingeblendet werden
'''
plt.plot(x,y)
plt.plot(x,y_smoothed, color='red')
plt.xlabel('Frame')
plt.ylabel('Höhe')
plt.show()
'''

minima_log = []
for x in range(0,len(maxima_log)-1, 1):
    minima_log.append(int((maxima_log[x] + maxima_log[x+1])/2))

zwischenpunkte_log = []

for x in range(0,len(maxima_log)-1, 1):
    maximum = y_smoothed[maxima_log[x + 1]]
    minimum = y_smoothed[minima_log[x]]
    searched_angle = int((maximum + minimum) / 2)
    try:
        zwischenpunkt = takeClosest(y_smoothed, searched_angle, minima_log[x] ,maxima_log[x+1]) #um Bild mit richtigem Winkel
    except:
        break
    zwischenpunkte_log.append(zwischenpunkt)
for x in range(0,len(maxima_log)-1, 1):
    maximum = y_smoothed[maxima_log[x]]
    minimum = y_smoothed[minima_log[x]]
    searched_angle = int((maximum + minimum) / 2)
    try:
        zwischenpunkt = takeClosest(y_smoothed, searched_angle, maxima_log[x], minima_log[x])
    except:
        break
    zwischenpunkte_log.append(zwischenpunkt)

writing_list = [] #Liste mit Framenummern dessen zugehoerige Frames abgespeichert werden sollen

for x in zwischenpunkte_log:
    writing_list.append(x)
for y in maxima_log:
    writing_list.append(y)
for z in minima_log:
    writing_list.append(z)
writing_list.sort() #Framenummern sortieren

for i in writing_list: #Bilder für chosen_pics aussuchen
    cap = cv2.VideoCapture(path)
    cap.set(1, i)  #i steht für Framenummer welche Geladen werden soll
    ret, img = cap.read()
    #wenn man die abzuspeichernden Bilder skalieren möchte, müssen folgende ausgeklammerte Zeilen eingeblendet werden
    '''
    if img.shape[0] > 4000:
        img = cv2.resize(img, (int(img.shape[1] / 7), int(img.shape[0] / 7)))
    elif img.shape[0] > 2000:
        img = cv2.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
    elif img.shape[0] > 1000:
        img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    '''
    cv2.imwrite(r'C:\...\frame{}.jpg'.format(i), img) #abspeichern

#Falls der Bewegungsverlauf des Markers abgespeichert werden soll muss folgendes eingeblendet werden
#'''
if img.shape[0] > 4000:
    img = cv2.resize(img, (int(img.shape[1] / 7), int(img.shape[0] / 7)))
elif img.shape[0] > 2000:
    img = cv2.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
elif img.shape[0] > 1000:
    img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))

im = create_blank(img.shape[1], img.shape[0])
a = np.array(point_list)
cv2.drawContours(im, [a], -1, (0, 255, 0), 2)
cv2.imwrite(r'C:\...\frames_Marker_history.jpg', im)
#'''