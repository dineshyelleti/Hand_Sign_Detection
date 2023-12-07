import cv2
import mediapipe as mp
import numpy as np
import time

class handDetector():

    def __init__(self,mode=False,maxHands=2,modelC=1,minDetectionConfi=0.5,minTrackConfi=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.minDetectionConfi = minDetectionConfi
        self.minTrackConfi = minTrackConfi

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelC,self.minDetectionConfi,self.minTrackConfi)
        self.mpDraw = mp.solutions.drawing_utils

    
    def findHands(self, frame, draw_lines=True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #   print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in (self.results.multi_hand_landmarks):
                if draw_lines:
                    landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    connection_drawing_spec = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS,
                                               landmark_drawing_spec,connection_drawing_spec)
                else:
                    self.mpDraw.draw_landmarks(frame, handLms)
        return frame
    
    def findPosition(self, frame, handno=0, draw_lines=True):
        
        landmark_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h,w,c = frame.shape
                pixel_x, pixel_y = int(lm.x * w), int(lm.y * h)
                #print(id, pixel_x, pixel_y)
                landmark_list.append([id, pixel_x, pixel_y])
                if draw_lines:
                    cv2.circle(frame,(pixel_x, pixel_y), 3, (255,0,255), -1)
        return landmark_list 
        


def main():
    
    cap = cv2.VideoCapture(0)

    ctime = 0
    ptime = 0

    detector = handDetector()
    while True:

        ret, img = cap.read()

        img = detector.findHands(img,draw_lines = True)
        lmlist = detector.findPosition(img)
        # if len(lmlist) != 0 :
        #     print(lmlist[4])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3, cv2.LINE_AA)
        ptime = ctime

        cv2.imshow("Hand_Tracking", img)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()