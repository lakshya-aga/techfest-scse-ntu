"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import time
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

data=[]

for i in range(1,11):
    time.sleep(1)
    emotion=""
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
    
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
    
        frame = gaze.annotated_frame()
        text = ""
    
        if gaze.is_blinking():
            emotion = "a"
            text = "Blinking"
        if gaze.is_right() or gaze.is_left():
            emotion="not attentive"
            text="not attentive"
        elif gaze.is_center():
            emotion = "attentive" 
            text="attentive"
    
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
        cv2.imshow("Demo", frame)
    
        if emotion!="":
            break
    if(emotion=="attentive" or emotion=="not attentive"):
        data.append([1*i,emotion])

print(data)
webcam.release()
cv2.destroyAllWindows()