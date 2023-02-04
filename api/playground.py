import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
#from tensorflow.keras import models, layers



file = "../prova2.jpg"
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
image = Image.open(file)
image = image.resize([48, 48])
image = np.array(image)/255
image = tf.image.rgb_to_grayscale(image)
iamge_batch = np.expand_dims(image, 0)

model = tf.keras.models.load_model("../models/model2")
predictions = model.predict(iamge_batch)
confidence = dict(zip(class_names, predictions[0]))


#%%
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


values = get_prediction(picture)
#fig = plot_prediction(values)
st.write(values)

#st.plotly_chart(fig, use_container_width=True)




#%%
"""
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
values = get_prediction("../prova2.jpg")
plot_prediction(values)


#%%
