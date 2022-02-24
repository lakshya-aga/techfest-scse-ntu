import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
model=model_from_json(open(r"D:\jinda\python programs\techfest\fer.json",'r').read())
model.load_weights(r"D:\jinda\python programs\techfest\fer.h5")
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')


cap= cv2.VideoCapture(0)

st.header("This is working")

while True:
    check, frame = cap.read()
    if not check:
        break
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(grey,1.3,3)
    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,100,50),4)
        roi_grey=grey[y:y+w,x:x+h]
        roi_grey=cv2.resize(roi_grey,(48,48))
        img_pixels=image.img_to_array(roi_grey)
        img_pixels=np.expand_dims(img_pixels,axis=0)
        img_pixels/=255
        predictions=model.predict(img_pixels)
        max_index=np.argmax(predictions[0])
        emotions=('Angry','','Fear','Happy','Sad','Confused','Neutral')
        predicted_emotions=emotions[max_index]
        cv2.putText(frame,predicted_emotions,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    resized_img=cv2.resize(frame,(1000,700))
    cv2.imshow('video',resized_img)
    key = cv2.waitKey(10) 
    if(key==ord('q')):
        break
    

cap.release()
cv2.destroyAllWindows()

