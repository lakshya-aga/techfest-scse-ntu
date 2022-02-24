import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
from gaze_tracking import GazeTracking
import time
import pandas as pd
import matplotlib.pyplot as plt

model=model_from_json(open(r"D:\jinda\python programs\techfest\fer.json",'r').read())
model.load_weights(r"D:\jinda\python programs\techfest\fer.h5")
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')



st.header("PROJECT TITLE")

choice=st.radio("What would you like to do?", ("Home Screen","Measure Engagement", "Measure Attention"))



if(choice=="Measure Engagement"):
    
    cap= cv2.VideoCapture(0)
        
    def OccurEmot(input_emot):
        global data
        timestamps=[]
        for i in range(len(data)):
            if(input_emot==data[i][1]):
                timestamps.append(data[i][0])
        return [timestamps, len(timestamps)]
    
    emot=["Happy", "Sad", "Confused", "Neutral"]
    
    data=[]
    
    st.header("")
    st.subheader("Processing Facial Cues")
    st.write("This function allows you to estimate the engagement of an audience")
    st.header("")
    
    for i in range(1,11):
        time.sleep(1)
        emotion=""
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
                emotion = predicted_emotions
                cv2.putText(frame,predicted_emotions,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            resized_img=cv2.resize(frame,(1000,700))
            cv2.imshow('video',resized_img)
            #key = cv2.waitKey(10) 
            if(emotion!=""):
                break
        data.append([1*i, emotion])
    
    
    df=pd.DataFrame(data, columns=["Time", "Emotion"])
    
    emot1_info=OccurEmot("Happy")
    emot2_info=OccurEmot("Sad")
    emot3_info=OccurEmot("Confused")
    emot4_info=OccurEmot("Neutral")
    
    st.header("")
    st.subheader("Below are the indicators for different emotions")
    st.header("")
    
    st.subheader("Happy Emotion")
    emot1_status=[]
    for i in range (1,11):
        c=0
        for j in range (len(emot1_info[0])):
            if(i==emot1_info[0][j]):
                c+=1
        if(c==0):
            emot1_status.append(0)
        else:
            emot1_status.append(1)
    fig = plt.figure(figsize = (10, 5))
    plt.plot([1,2,3,4,5,6,7,8,9,10], emot1_status)
    plt.xlabel("Time")
    plt.ylabel("True/False")
    plt.title("Happy Expression")
    st.pyplot(fig)
    
    st.header("")
    st.subheader("Sad Emotion")
    emot2_status=[]
    for i in range (1,11):
        c=0
        for j in range (len(emot2_info[0])):
            if(i==emot2_info[0][j]):
                c+=1
        if(c==0):
            emot2_status.append(0)
        else:
            emot2_status.append(1)
    fig = plt.figure(figsize = (10, 5))
    plt.plot([1,2,3,4,5,6,7,8,9,10], emot2_status)
    plt.xlabel("Time")
    plt.ylabel("True/False")
    plt.title("Sad Expression")
    st.pyplot(fig)
    
    st.header("")
    st.subheader("Confused Emotion")
    emot3_status=[]
    for i in range (1,11):
        c=0
        for j in range (len(emot3_info[0])):
            if(i==emot3_info[0][j]):
                c+=1
        if(c==0):
            emot3_status.append(0)
        else:
            emot3_status.append(1)
    fig = plt.figure(figsize = (10, 5))
    plt.plot([1,2,3,4,5,6,7,8,9,10], emot3_status)
    plt.xlabel("Time")
    plt.ylabel("True/False")
    plt.title("Confused Expression")
    st.pyplot(fig)
    
    st.header("")
    st.subheader("Neutral Emotion")
    emot4_status=[]
    for i in range (1,11):
        c=0
        for j in range (len(emot4_info[0])):
            if(i==emot4_info[0][j]):
                c+=1
        if(c==0):
            emot4_status.append(0)
        else:
            emot4_status.append(1)
    fig = plt.figure(figsize = (10, 5))
    plt.plot([1,2,3,4,5,6,7,8,9,10], emot4_status)
    plt.xlabel("Time")
    plt.ylabel("True/False")
    plt.title("Neutral Expression")
    st.pyplot(fig)
    
    st.header("")
    st.subheader("Collated data in a Pie Chart")
    emot1per=emot1_info[1]*100/10
    emot2per=emot2_info[1]*100/10
    emot3per=emot3_info[1]*100/10
    emot4per=emot4_info[1]*100/10
    
    label1=["Happy", "Sad", "Confused", "Neutral"]
    size1=[emot1per, emot2per, emot3per, emot4per]
    fig1,ax1=plt.subplots()
    ax1.pie(size1, labels=label1, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)
    
    
    
    cap.release()
    cv2.destroyAllWindows()

if(choice=="Measure Attention"):
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    data=[]
            
    count_a=0
    count_na=0
    
    st.header("")
    st.subheader("Processing Facial Cues")
    st.write("This function allows you to measure what percentage of time you were attentive/inattentive")
    st.header("")
    
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        if gaze.is_right() or gaze.is_left():
            count_na+=1
            text="not attentive"
        elif gaze.is_center():
            emotion = "attentive" 
            count_a+=1

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1)==27:
            break


    data=[['attentive',count_a],['not attentive',count_na]]
    attpec=count_a*100/(count_a+count_na)
    nattpec=100-attpec
    label2=["Attentive", "Inattentive"]
    size2=[attpec, nattpec]
    fig2,ax2=plt.subplots()
    ax2.pie(size2, labels=label2, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)
    webcam.release()
    cv2.destroyAllWindows()
    