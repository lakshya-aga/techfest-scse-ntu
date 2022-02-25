#loading in data from csv file
import csv
from google.colab import drive
import numpy as np
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
import datetime
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation , Dropout ,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import cv2

#loading the data from csv file
drive.mount('/content/drive')
train_label2=[]
train_image2=[]
test_label2=[]
test_image2=[]
f=open("/content/drive/My Drive/Colab_Notebooks/newfer.csv","rt")
data = csv.reader(f)
for row in data:
  if(row[0].isdigit()):
    real_pixels=[]
    pixels=row[1].split(" ")
    for i in pixels:
      real=int(i)
      real_pixels.append(real)
    if(row[2]=="Training"):
      train_label2.append(row[0])
      train_image2.append(real_pixels)
    elif(row[2]=="PublicTest"):
      test_label2.append(row[0])
      test_image2.append(real_pixels)
    else:
      continue
f.close()

#reshaping the data as required by model
train_image2=np.array(train_image2)
test_image2=np.array(test_image2)
train_image2=train_image2.reshape(train_image2.shape[0],48,48,1)
test_image2=test_image2.reshape(test_image2.shape[0],48,48,1)
train_image2=train_image2/255.0
test_image2=test_image2/255.0

test_label_2=[]
for i in test_label2:
  i=int(i)
  test_label_2.append(i)
train_label_2=[]
for i in train_label2:
  i=int(i)
  train_label_2.append(i)

train_label_2=np.array(train_label_2)
test_label_2=np.array(test_label_2)
test_label_2=to_categorical(test_label_2,num_classes=7)
train_label_2=to_categorical(train_label_2,num_classes=7)

#creating the model
with tf.device('/gpu:0'):
  model=Sequential()

  model.add(Conv2D(32,(3,3),input_shape=(48,48,1),activation='relu',padding="same"))
  model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
  model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
  model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
 

  model.add(Flatten(input_shape=(48,48)))

  model.add(Dense(64,activation='relu'))
  

  model.add(Dropout(0.3))
  model.add(Dense(7,activation='sigmoid'))

#running and saving the model
logdir = os.path.join("logs", datetime.datetime.now().strftime("6c_2d_128bs_sigmoid"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
  
model.fit(x=train_image2,
            y=train_label_2,
            epochs=10,
            validation_data=(test_image2,test_label_2),
            callbacks=[tensorboard_callback],
            batch_size=128)

fer_json = model.to_json()
with open("/content/drive/My Drive/Colab_Notebooks/fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("/content/drive/My Drive/Colab_Notebooks/fer.h5")

        

