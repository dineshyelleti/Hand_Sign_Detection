import cv2
import mediapipe as mp
import numpy as np
import time

class handDetector():

    def __init__(self,mode=False,maxHands=2,modelConfi=1,minDetectionConfi=0.5,minTrackConfi=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.modelConfi = modelConfi
        self.minDetectionConfi = minDetectionConfi
        self.minTrackConfi = minTrackConfi

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelConfi,self.minDetectionConfi,self.minTrackConfi)
        self.mpDraw = mp.solutions.drawing_utils

    
    def findHands(self, frame, draw_lines=True, draw_box=True, label=True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        all_Hands_info = []
        #   print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms, handType in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                Hand_info = {}
                lmList = []
                xList = []
                yList = []
                for id, lms in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    pixel_x, pixel_y, pixel_z = int(w * lms.x), int(h * lms.y), int(w * lms.z)
                    lmList.append([id, pixel_x, pixel_y, pixel_z])
                    xList.append(pixel_x)
                    yList.append(pixel_y)
                
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                cx, cy = (xmin + xmax) // 2 , (ymin + ymax)//2
                hand_label = "Left" if handType.classification[0].label == 'Right' else "Right"

                Hand_info['type'] = hand_label
                Hand_info['landmarks_List'] = lmList
                Hand_info['boundary_box'] = [xmin,xmax,ymin,ymax]
                Hand_info['center'] = (cx,cy)

                all_Hands_info.append(Hand_info)                

                if label:
                    cv2.putText(frame, hand_label, (xmin-20,ymin-30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                if draw_box:
                    cv2.rectangle(frame, (xmin-20,ymin-20), (xmax+20,ymax+20), (255,0,0), 2)                   

                if draw_lines:
                    landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    connection_drawing_spec = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS,
                                               landmark_drawing_spec,connection_drawing_spec)
                else:
                    self.mpDraw.draw_landmarks(frame, handLms)
        return all_Hands_info,frame
    
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

        all_hand_info, img = detector.findHands(img,draw_lines = True,draw_box=False,label=True)
        lmlist = detector.findPosition(img)
        # if len(lmlist) != 0 :
        #     print(lmlist[4])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3, cv2.LINE_AA)
        ptime = ctime

        cv2.imshow("Hand_Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f'BTP PROJECT\Hand_Sign_Detection\Image.jpg', img)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()