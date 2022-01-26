import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
videoObjcap = cv2.VideoCapture(0)


while(True):
    cap , frame = videoObjcap.read()
    scale = 10
    width = int(frame.shape[1]*20/100)
    height = int(frame.shape[0]*20/100)
    dim = (width,height)
    resized = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor= 1.1,minNeighbors= 5,minSize=(10, 10)) 
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 255, 0), 2)
    crop = resized[y:y+h, x:x+w]
    cv2.imwrite('facetest.jpg',crop)
    crop = cv2.resize(crop,(64,64))
    X_test = np.asarray(crop)
    model = keras.models.load_model("model.h5")
    X_test=X_test.reshape(-1,64,64,3)
    predictions = model.predict(X_test)
    if(predictions[0]<=0.5):
        pred="Myron"
    else:
        pred=""
    cv2.putText(resized,pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Capturing Frames',resized)
    
    
    print(predictions)
    if(cv2.waitKey(1) & 0xFF == ord('s')):
        videoObjcap.release()
        break
    print("Success")
    
videoObjcap.release()
cv2.destroyAllWindows()