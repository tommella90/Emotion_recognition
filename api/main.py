import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/model2")
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@app.get("/ping")
async def ping():
    return "stocazzo"

def read_files_as_img(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    #image = tf.image.resize(image, [48, 48])/255
    #image = tf.io.read_file(data)
    #image = tf.image.decode_jpeg(image)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_files_as_img(await file.read())  # await is to wait for the file to be read
    iamge_batch = np.expand_dims(image, 0)
    #image = tf.io.read_file(file)
    #image = tf.image.decode_jpeg(image)
    #image = tf.image.resize(image, [48, 48])/255
    #image = read_files_as_img(file)
    predictions = MODEL.predict(iamge_batch)
    #confidence = dict(zip(CLASS_NAMES, prediction[0]))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class, confidence)
    dict_output = {
        "class_name": predicted_class,
        "confidence": confidence,
    }
    return dict_output


#%%
if __name__ == "__main__":
    uvicorn.run(app, port=7000, host="localhost")


#%%
x = predict("../prova2.jpg")
#%%
