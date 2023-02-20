from fastapi import FastAPI, File, UploadFile
from deta import Drive
from enum import Enum
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import matplotlib.pyplot as plt
import os
import uvicorn


#%%
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

with open('models/network_emotions.json', 'r') as json_file:
    emo_model = json_file.read()
network_loaded = tf.keras.models.model_from_json(emo_model)
network_loaded.load_weights('models/weights_emotions.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

def load_image(file):
    image = cv2.imread(file)
    return image

def predict_emotion_from_image(image):
    original_image = image.copy()
    faces = face_detector.detectMultiScale(original_image, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    model_size = 48
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (model_size, model_size))
    roi = roi/255
    roi = np.expand_dims(roi, axis=0)
    probs = network_loaded.predict(roi)
    predicted = class_names[np.argmax(probs)]
    return probs, predicted

image = load_image("prova2.jpg")
probs, predicted = predict_emotion_from_image(image)


#%%
app = FastAPI()
files = Drive("myfiles")


@app.get("/allora")
async def hello():
    return "welcome"

@app.post("/")
def upload_image(file: UploadFile = File(...)):
    return files.put(file.filename, file.file)

@app.post("/api/predict_emotion")
def predict_emotion(file: UploadFile = File(...) ):

    image = load_image(file)
    #predictions = predict_emotion_from_image(image)
    return image


if __name__ == "__main__":
    uvicorn.run(app, port=8888, host='127.0.0.1')

#%%

#%% 2) TENSORFLOW SERVICE
# docker pull tensorflow/serving
# https://github.com/tensorflow/serving

# enter the docker image
#docker run -it -v C:\Users\tomma\Documents\data_science\ml_pj\emotion_recognition\models:/tf_serving -p 8888:8888 --entrypoint /bin/bash tensorflow/serving
""" RUN THE DOCKER IMAGE
- docker run folder_with_models
- :/tf_serving -p 8888:8888 --entrypoint /bin/bash (bo)
- docker image from tf
"""
# ls -ltr tf_serving to see models saved in the dir
# tensorflow_model_server --rest_api_port=8888 --model_name=model_test --model_base_path=/tf_serving
"""
- tensorflow_model_server --rest_api_port=PREVIOUS_PORT
- --model_name=CHOOSE_NAME
- --model_base_path=/MY_DIR
"""

#go on http://localhost:8888/v1/models/model_test

