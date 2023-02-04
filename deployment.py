from fastapi import FastAPI
from enum import Enum
import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import matplotlib.pyplot as plt
import os
import uvicorn

#%% 1) FASTAPI
app = FastAPI()

model = tf.keras.models.load_model("models/model2")
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_image(file, image_size=[48, 48]):
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, image_size)/255
    #img = tf.image.rgb_to_grayscale(img)
    return img

def get_prediction(img):
    pred = model.predict(img)
    confidence = dict(zip(class_names, pred[0]))
    class_index = np.argmax(confidence)
    class_name = class_names[class_index]
    return confidence

#img = preprocess_image("prova2.jpg")
#pred = get_prediction(img)

#%%
def main(file):
    img = preprocess_image(file)
    return get_prediction(img)


#%%
@app.get("/get_emotion_prediction/{file}")
def get_prediction(img):
    img = preprocess_image(f"{img}")
    pred = get_prediction(img)
    return pred

# uvicorn fast_api_tutorial:app --reload



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
