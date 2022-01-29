#!/usr/bin/env python
# coding: utf-8

# In[60]:


import tensorflow
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import models
from typing import List

import json

import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

import PIL
from PIL import ImageFilter

from matplotlib.pyplot import imshow

import uvicorn
from fastapi import FastAPI, File, UploadFile
import io

from io import BytesIO

from zipfile import ZipFile

import cv2



# In[61]:


def load_model():
    #zf = ZipFile("models/top_model.h5.zip")# Extract its contents into
    #zf.extractall(path = "models/")
        # close the ZipFile instance
    #zf.close()
    
    model = keras.models.load_model("models/top_model.h5")
    print("Model loaded")
    return model

def read_image(bin_data, size=(105, 105)):
    """Load image

    Arguments:
    bin_data {bytes} --Image binary data

    Keyword Arguments:
    size {tuple} --Image size you want to resize(default: {(224, 224)})

    Returns:
    numpy.array --image
    """
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    return img


def predict(img):
    model = load_model()
    #pil_im = PIL.Image.open(img_path).convert("L")
    #pil_im = pil_image(img_path)
    org_img = img_to_array(img)
    predictions = model.predict(np.asarray(org_img, dtype="float").reshape([1, 105, 105, 1]) / 255.0)
    probas = []
    keys = ["Amatic", "Arial", "Calibri", "Cambria", "Cambriab", "Caveat", "Comfortaa", "ComicSansMS", "Consolas", "CourierNew", "DroidSans", "DroidSerif", "FreeSans", "Lora", "Merriweather", "MonotypeCorsiva", "Montserrat", "Nunito", "Roboto", "TimesNewRoman", "Ubuntu", "UbuntuMono", "Unicode"]
    for key, value in zip(keys, predictions[0]):
        probas.append({"font":key, "probability":value})
    
    result = {}
    result["data"] = {}
    result["data"]["probabilities"] = probas
    result["status"] = "OK"
    
    return result


# In[55]:
app = FastAPI()


@app.post("/prediction")
async def predict_api(files: List[UploadFile] = File(...)):
    #extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    bin_data = io.BytesIO(files[0].file.read())
    img = read_image(bin_data)
    #if not extension:
       # return "Image must be jpg or png format!"
    #image = read_imagefile(await file.read())
    prediction = predict(img)
    return json.loads(str(prediction).replace("'", '"'))


@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, debug=True, host='0.0.0.0')


# In[ ]:




