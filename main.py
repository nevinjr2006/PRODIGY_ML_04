import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
data_dir="/path/to/leapGestRecog"
gestures=os.listdir(data_dir)
img_size=64
data=[]
labels=[]
for gesture in gestures:
    gesture_path=os.path.join(data_dir,gesture)
    for person in os.listdir(gesture_path):
        person_path=os.path.join(gesture_path,person)
        for img_file in os.listdir(person_path):
            img_path=os.path.join(person_path,img_file)
            img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(img_size,img_size))
            data.append(img)
            labels.append(gesture)
X=np.array(data).reshape(-1,img_size,img_size,1)/255.0
lb=LabelBinarizer()
y=lb.fit_transform(labels)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=Sequential([Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,1)),MaxPooling2D(2,2),
Conv2D(64,(3,3),activation='relu'),MaxPooling2D(2,2),Flatten(),Dense(128,activation='relu'),Dropout(0.5),
Dense(10,activation='softmax')])            
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history=model.fit(X_train,y_train,epochs=10,validation_split=0.1,batch_size=32)
loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test Accuracy:{accuracy:.2f}")
import kagglehub
path = kagglehub.dataset_download("gti-upm/leapgestrecog")
print("Path to dataset files:", path)