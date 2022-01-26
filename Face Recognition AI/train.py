import numpy as np
import os
import glob
import cv2

import tensorflow as tf
data = []
img_dir = "/Users/myrondza/iCloud/com~apple~CloudDocs/Face\ Recognition/" 
files = glob.glob("*.jpg") 
X_train = [] 
for f1 in files: 
    img = cv2.imread(f1,-1) 
    print(img.shape)
    img = cv2.resize(img, (64,64))
    X_train.append(img)

print(files)
X_train = np.asarray(X_train)

print(X_train.shape)

y_train = np.array([0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])

sgd = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.1, nesterov=True)

def cnn_model() :   
    model = tf.keras.Sequential()
    

    model.add(tf.keras.layers.Conv2D(32,kernel_size=2, activation=tf.nn.relu, input_shape=(64,64,3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Conv2D(64,kernel_size=2,activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Conv2D(128,kernel_size=2,activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(256,kernel_size=2,activation=tf.nn.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-6),metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    return model

model = cnn_model()
model.fit(X_train, y_train, epochs=25,batch_size=10)
model.save('model.h5')