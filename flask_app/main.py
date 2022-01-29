#!/usr/bin/env python
# coding: utf-8

# In[60]:


import tensorflow
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import models

import json

import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

import PIL
from PIL import ImageFilter

from matplotlib.pyplot import imshow

import uvicorn
from fastapi import FastAPI, File, UploadFile

from io import BytesIO



# In[61]:


def load_model():
    model = keras.models.load_model("models/top_model.h5")
    print("Model loaded")
    return model

def pil_image(img_path):
    pil_im = PIL.Image.open(BytesIO(img_path)).convert("L")
    pil_im = pil_im.resize((105,105))
    return pil_im


def predict(img_path):
    model = load_model()
    #pil_im = PIL.Image.open(img_path).convert("L")
    pil_im = pil_image(img_path)
    org_img = img_to_array(pil_im)
    predictions = model.predict(np.asarray(org_img, dtype="float").reshape([1, 105, 105, 1]) / 255.0)
    probas = {}
    keys = ["Amatic", "Arial", "Calibri", "Cambria", "Cambriab", "Caveat", "Comfortaa", "ComicSansMS", "Consolas", "CourierNew", "DroidSans", "DroidSerif", "FreeSans", "Lora", "Merriweather", "MonotypeCorsiva", "Montserrat", "Nunito", "Roboto", "TimesNewRoman", "Ubuntu", "UbuntuMono", "Unicode"]
    for key, value in zip(keys, predictions[0]):
        probas[key] = str(value)
    
    result = {}
    result["data"] = probas
    result["status"] = "0"
    
    return str(result).replace("'", '"')


# In[55]:


app = FastAPI()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    #image = read_imagefile(await file.read())
    prediction = predict(await file.read())
    return json.loads(prediction)


@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, debug=True)


# In[ ]:




