from io import BytesIO
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow import keras
from tensorflow.keras import models
model = None
def load_model():
    model = keras.models.load_model("models/top_model.h5")
    print("Model loaded")
    return model
def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()
    image = np.resize(np.asarray(image), (105, 105))
    image = np.expand_dims(image, 0)
    image = image / 255
    #result = decode_predictions(model.predict(image), 2)[0]
    predictions = model.predict(image)
    probas = []
    keys = ["Amatic", "Arial", "Calibri", "Cambria", "Cambriab", "Caveat", "Comfortaa", "ComicSansMS", "Consolas", "CourierNew", "DroidSans", "DroidSerif", "FreeSans", "Lora", "Merriweather", "MonotypeCorsiva", "Montserrat", "Nunito", "Roboto", "TimesNewRoman", "Ubuntu", "UbuntuMono", "Unicode"]
    for key, value in zip(keys, predictions[0]):
        probas.append({"font":key, "probability":value})
    result = {}
    result["data"] = {}
    result["data"]["probabilities"] = probas
    result["status"] = "OK"
    return result
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(np.asarray(image)[...,:3], rgb_weights)
    return grayscale_image
app = FastAPI(title='Tensorflow FastAPI Starter Pack')
@app.post("/prediction")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    image = read_imagefile(await file.read())
    prediction = predict(image)
    print(prediction)
    return json.loads(str(prediction).replace("'", '"'))
if __name__ == "__main__":
    uvicorn.run(app, debug=True, host='0.0.0.0')
