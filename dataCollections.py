import cv2
import numpy as np
import mediapipe
import time
import Hand_Tracking_Module_Intermediate as htmi
import math
#from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
folder = "BTP PROJECT\Hand_Sign_Detection\Data_Collected\H"
counter = 1

detector = htmi.handDetector(maxHands=1)
boundary_offset = 20

while True:
    success, img = cap.read()
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

                cv2.imshow("White Screen", white_screen)
                cv2.imshow("Cropped Image", hand_img_crop)


        except Exception as e:
            print("Box out of boundary")

        cv2.imshow("Web_Cam",img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            print(counter)
            cv2.imwrite(f'BTP PROJECT\Hand_Sign_Detection\Image.jpg', hand_img_crop)
            counter += 1
        elif key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
