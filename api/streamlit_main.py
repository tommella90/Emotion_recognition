from __future__ import absolute_import, division, print_function, unicode_literals
import streamlit as st
import tensorflow as tf
import urllib.request
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import plotly.graph_objects as go
import os
from tensorflow.keras.utils import plot_model


class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
face_detector = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

with open('../models/network_emotions.json', 'r') as json_file:
    emo_model = json_file.read()
network_loaded = tf.keras.models.model_from_json(emo_model)
network_loaded.load_weights('../models/weights_emotions.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="CREATE WORD CLOUDS",
                   page_icon=":🧊:")

with st.container():
    st.title("TITLE")
    st.subheader("selfieee")
    st.write(network_loaded.summary())


#%%
"""
def plot_prediction(values):
    fig = go.Figure(data=go.Scatterpolar(
        r=list(values.values()),
        theta=list(values.keys()),
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False
    )

    fig.show()



with st.container():
    st.title("TITLE")
    st.subheader("selfieee")

    picture = st.camera_input("selfiee",
                                    key=None,
                                    help=None,
                                    on_change=None,
                                    args=None,
                                    kwargs=None,
                                    disabled=False,
                                    label_visibility="visible")


    if picture:
        st.image(picture, caption="selfieee", width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
        #values = get_prediction(picture)
        #fig = plot_prediction(values)
        #st.write(values)

#st.plotly_chart(fig, use_container_width=True)

with st.container():
    st.title("pred")
    st.write(get_prediction(picture))



def get_prediction(image):
    image = Image.open(image)
    image = image.resize([48, 48])
    image = np.array(image)/255
    image = tf.image.rgb_to_grayscale(image)
    iamge_batch = np.expand_dims(image, 0)

    model = tf.keras.models.load_model("../models/model2")
    predictions = model.predict(iamge_batch)
    confidence = dict(zip(class_names, predictions[0]))
    return confidence

def plot_prediction(values):
    fig = go.Figure(data=go.Scatterpolar(
        r=list(values.values()),
        theta=list(values.keys()),
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False
    )

    fig.show()

"""

#%%
