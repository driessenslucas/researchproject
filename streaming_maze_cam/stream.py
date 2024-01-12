import cv2
import warnings
warnings.filterwarnings('ignore')    

cap = cv2.VideoCapture("http://192.168.0.25:8500/?action=stream")

while True:

    ret, frame = cap.read()
    cv2.imshow('video', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
