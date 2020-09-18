# It is a real time Rock Paper Scissors game with a camera
# Scissors or rock or paper in the area
# Start: Press space bar
# The result of the computer: random

import cv2
import numpy as np
import math
from datetime import datetime
import random

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize variables
camSource = 0
running = True
saveCount = 0
nSecond = 0
totalSec = 3
strSec = '321'
keyPressTime = 0.0
startTime = 0.0
timeElapsed = 0.0
startCounter = False
endCounter = False
press = 0

usrwin = 0
comwin = 0

frameWidth = 480
frameHeight = 640


def com_Rock_Scissors_paper():
    RCP = ['Rock', 'Scissors', 'Paper']
    random.shuffle(RCP)
    return RCP[0]


def Rock_Scissors_paper(frame):
    # The region of a hand
    roi = frame[50:300, 400:640]
    cv2.imshow("1", roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Hand region
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Binary extraction
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise cancellation
    mask = cv2.medianBlur(mask, 9)

    cv2.imshow("Afer Filter", mask)

    # closing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=5)

    cv2.imshow("Afer Closing", mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(roi, contours,-1,(0,255,0),3)

    cv2.imshow("Contours1", roi)

    # Exception: There is a no hand1
    if not contours:
        return 'Nothing'

    
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull) 

    # convexity Defects counts
    l = 0
     # Exception: There is a no hand2
    if defects is None:
        return 'Nothing'
    # Iteration with points
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

        d = (2 * ar) / a  

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        
        if angle <= 90 and d > 45:
            l += 1
            cv2.circle(roi, far, 3, [255, 0, 0], -1)

        # draw lines around hand
        cv2.line(roi, start, end, [0, 255, 0], 2)

    # Separating Rock Paper Scissors
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l == 0:

        cv2.putText(frame, 'Rock', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        return 'Rock'

    elif l == 1:
        cv2.putText(frame, 'Scissors', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        return 'Scissors'

    else:
        cv2.putText(frame, 'Paper', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        return 'Paper'


def fight(user, computer):
    global comwin
    global usrwin
    if user == computer:
        print('Draw')

    elif (user == 'Rock' and computer == 'Scissors') or (user == 'Scissors' and computer == 'Paper') or (
            user == 'Paper' and computer == 'Rock'):
        print('win')
        usrwin += 1
    elif user == 'Nothing':
        print('Put hand in the box')
    else:
        print('Defeat')
        comwin += 1


mask = np.zeros((100, 80, 3), np.uint8)
while (1):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[50:300, 400:640]
    cv2.rectangle(frame, (400, 50), (640, 300), (0, 255, 0), 0)

    cv2.putText(frame, 'Win:' + str(usrwin), (0, 300), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, 'Defeat:' + str(comwin), (0, 350), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    user_result = Rock_Scissors_paper(frame)

    if press == 0:
        cv2.putText(frame, 'Press Space Bar', (0, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    if startCounter:
        press = 1
        if nSecond < totalSec:
            cv2.putText(img=frame,
                        text=strSec[nSecond],
                        org=(int(frameWidth / 2 - 20), int(frameHeight / 2)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=6,
                        color=(255, 255, 255),
                        thickness=5,
                        lineType=cv2.LINE_AA)

            timeElapsed = (datetime.now() - startTime).total_seconds()

            if timeElapsed >= 1:
                nSecond += 1
                timeElapsed = 0
                startTime = datetime.now()
        
        else:
            user_result = Rock_Scissors_paper(frame)
            com_result = com_Rock_Scissors_paper()
            print("--------------------")
            print("User: " + user_result)
            print("Computer: " + com_result)

            fight(user_result, com_result)
            print("--------------------")
            saveCount += 1
            startCounter = False
            nSecond = 0
            press = 0

    k = cv2.waitKey(3) & 0xFF
    if k == 27:
        break
    elif k == 32:
        startCounter = True
        startTime = datetime.now()

    cv2.imshow('frame', frame)

cv2.destroyAllWindows()
cap.release()
