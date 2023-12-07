import cv2
import numpy as np
import mediapipe
import time
import Hand_Tracking_Module_Intermediate as htmi
import math
from Image_Classifier import Classifier
import tensorflow

cap = cv2.VideoCapture(0)
classifier = Classifier("Model\keras_model.h5","Model\labels.txt")
detector = htmi.handDetector(maxHands=1)
boundary_offset = 20

labels = ["A","B","C","D","E","F","G","H"]

curtime = 0
prevtime = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    if success:
        hands_info, img = detector.findHands(img)

        try :
            if hands_info:
                hand1 = hands_info[0]
                xmin,xmax,ymin,ymax = hand1['boundary_box']
                h,w = ymax - ymin,xmax - xmin
                white_screen = np.ones((300,300,3), np.uint8)*255
                hand_img_crop = img[ymin-boundary_offset:ymax+boundary_offset, 
                                    xmin-boundary_offset:xmax+boundary_offset]
                aspect_ratio = h/w
                if aspect_ratio>1:
                    k = 300/h
                    wcal = math.ceil(k * w)
                    wgap = math.ceil((300 - wcal)/2)
                    cropImage_resize = cv2.resize(hand_img_crop,(wcal, 300))
                    white_screen[0:300,wgap:wgap+wcal] = cropImage_resize
                    
                else:
                    hcal = math.ceil(300*aspect_ratio)
                    hgap = math.ceil((300 - hcal)/2)
                    cropImage_resize = cv2.resize(hand_img_crop,(300, hcal))
                    white_screen[hgap:hgap+hcal,0:300] = cropImage_resize

                prediction, index = classifier.getPrediction(white_screen)

                cv2.rectangle(imgOutput, (xmin-boundary_offset-1,ymin-2*boundary_offset-5), 
                                (xmax+boundary_offset+1,ymin-boundary_offset), (255,0,0),-1)
                cv2.rectangle(imgOutput, (xmin-boundary_offset,ymin-boundary_offset), 
                                (xmax+boundary_offset,ymax+boundary_offset), (255,0,0), 2) 
                cv2.putText(imgOutput,labels[index],(xmin-20,ymin-20-2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

                cv2.rectangle(imgOutput, (393,0),(640,30), (255,0,0),-1,cv2.LINE_AA)
                cv2.putText(imgOutput,"Confi : "+str(prediction[index]*100)[:5]+"%",(395,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


        except Exception as e:
            print("Box out of boundary")

        curtime = time.time()
        fps = 1/(curtime-prevtime)
        cv2.rectangle(imgOutput, (0, 0, 150, 30), (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(imgOutput, "FPS : "+str(int(fps)), (5,25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2, cv2.LINE_AA)
        prevtime = curtime

        cv2.imshow("Web_Cam",imgOutput)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
