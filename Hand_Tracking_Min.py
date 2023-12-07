import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ctime = 0
ptime = 0

while True:

    ret, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in (results.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h,w,c = frame.shape
                pixel_x, pixel_y = int(lm.x * w), int(lm.y * h)
                print(id, pixel_x, pixel_y)


            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3, cv2.LINE_AA)
    ptime = ctime

    cv2.imshow("Hand_Tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows