import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


crop = cv2.imread('face_7.jpg')
crop = cv2.resize(crop,(64,64))
X_test = np.asarray(crop)
model = keras.models.load_model("model.h5")
X_test=X_test.reshape(-1,64,64,3)
predictions = model.predict(X_test)
print(predictions)
if(predictions[0]<=0.5):
        pred="Myron"
else:
        pred=""
print(pred)
print("Success")