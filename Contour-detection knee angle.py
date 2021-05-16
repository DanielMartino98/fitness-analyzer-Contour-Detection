import cv2
import numpy as np
import math
from scipy.signal import find_peaks
import os.path
import imutils
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time

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
lower_color = np.array([18, 50, 20], np.uint8) #gelb
higher_color = np.array([70, 255, 255], np.uint8) #ohne dunkelgrün
#Farbspektrum für blau
'''
#lower_color = np.array([80, 50, 50])  
#higher_color = np.array([130, 255, 255])
'''

#Falls die Bearbeitungszeit gestoppt werden soll, muss folgende Zeile eingeblendet werden
#t0 = time.time()

angle_log = []
angle_index_log = []
nummer = 0
write_centers = []
write_contours = []

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

    hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, lower_color, higher_color)
    cnts = cv2.findContours(frame_threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finde Konturen
    cnts = imutils.grab_contours(cnts)

    center = None

    found_points_list = []
    circularity_list = []
    contour_list = []

    for c in cnts:
    #Mittelpunkte der Konturen finden
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        try:
            Circularity = (4*3.14*area)/(perimeter)**2
        except:
            Circularity = 0

        if area > 40 and Circularity > 0.4:

            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            found_points_list.append((cX, cY))
            circularity_list.append(Circularity)
            contour_list.append(c)
        else:
            continue

    while len(found_points_list) > 3: #Falls mehr als 3 Punkte gefunden wurden entferne die 'Schlechtesten'
        index = circularity_list.index(min(circularity_list))
        found_points_list.pop(index)
        circularity_list.pop(index)
        contour_list.pop(index)

    write_centers.append(found_points_list)
    write_contours.append(contour_list)

    if len(found_points_list) < 3: #wenn weniger als 3 Marker entdeckt wurden, werden die Umrandungen und Schwerpunkte nicht eingezeichnet
        pass
    else:
        a = np.array(found_points_list)
        cv2.drawContours(frame, [a], 0, (255, 255, 255), 2)

        for i in range(3):
            cv2.drawContours(frame, [contour_list[i]], -1, (0, 255, 0), 2)
            cv2.circle(frame, found_points_list[0], 7, (255, 0, 0), -1)
            cv2.circle(frame, found_points_list[1], 7, (255, 0, 0), -1)
            cv2.circle(frame, found_points_list[2], 7, (255, 0, 0), -1)
            cv2.putText(frame, "P{}".format(i+1), (found_points_list[i][0] - 10, found_points_list[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    try:

        P1 = (found_points_list[0][0], found_points_list[0][1])
        P2 = (found_points_list[1][0], found_points_list[1][1]) # um P2 wird der winkel berechnet
        P3 = (found_points_list[2][0], found_points_list[2][1])

        if P2[0] < P3[0]:

            P3 = (found_points_list[1][0], found_points_list[1][1])
            P2 = (found_points_list[2][0], found_points_list[2][1])


        beta = np.rad2deg(math.atan2(abs(P2[0]-P1[0]), abs(P2[1]-P1[1])))
        gamma = np.rad2deg(math.atan2(abs(P2[0]-P3[0]), abs(P2[1]-P3[1])))

        angle = 180 - beta - gamma

        angle_log.append(angle)

        cv2.putText(frame, "Framenumber = {}".format(nummer), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, "Angle = {}".format("%.1f" %angle), (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    except:
        cv2.putText(frame, "Framenumber = {}".format(nummer), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "ERROR 404", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        try:
            angle_log.append(angle_log[-1])
        except:
            angle_log.append(0)

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

x = range(0, len(angle_log))
y = angle_log
y_smoothed = savgol_filter(y, 51, 3) #Kurve glätten

average = sum(y_smoothed)/len(y_smoothed) #Mittelwert der Liste ermitteln
maxima_log = find_peaks(y_smoothed, distance=50, height=(average, max(y_smoothed)))[0] #finde Maxima
#Falls die Kurve ausgegeben werden soll, müssen die folgenden Zeilen eingeblendet werden
'''
plt.plot(x,y)
plt.plot(x,y_smoothed, 'r--')
plt.xlabel('Frame / -', fontsize=14)
plt.ylabel('Höhe / Pixel', fontsize=14)
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

    if img.shape[0] > 4000:
        img = cv2.resize(img, (int(img.shape[1] / 7), int(img.shape[0] / 7)))
    elif img.shape[0] > 2000:
        img = cv2.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
    elif img.shape[0] > 1000:
        img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    im = img.copy()

    hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, lower_color, higher_color)

    a = np.array(write_centers[i])
    cv2.drawContours(im, [a], 0, (255, 255, 255), 2)

    for c in range(len(write_contours[i])):
            # draw the contour and center of the shape on the image
        cv2.drawContours(im, [write_contours[i][c]], -1, (0, 255, 0), 2)
        cv2.circle(im, write_centers[i][c], 7, (255, 0, 0), -1)
        cv2.putText(im, "P{}".format(c+1), (write_centers[i][c][0] - 20, write_centers[i][c][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


    cv2.putText(im, "Framenumber = {}".format(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(im, "Angle = {}".format("%.1f" % angle_log[i]), (20, 50), #damit nur eine Kommastelle steht
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imwrite(r'C:\...\frame{}.jpg'.format(i), im)
