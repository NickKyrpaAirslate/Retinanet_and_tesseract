from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# import keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import models

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import time
import math
#import albumentations as A
import pytesseract

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import warnings;
warnings.filterwarnings("ignore", message="numpy.dtype size changed");
warnings.filterwarnings("ignore", message="numpy.ufunc size changed");

tessdata_dir_config = r'--tessdata-dir "/usr/local/Cellar/tesseract/4.1.1/share/tessdata"'

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/resnet50_1.h5'

# Load your trained model

def load_model_weights(model_path):
# Load weights for model
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    return model


model = load_model_weights(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://localhost:5000/')


def model_elements(image, model):
  # Prepare test images to crop from
  crop = image.copy()
  crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
  
  # Define a dataframe  coordinates
  df = pd.DataFrame(columns = ['class_id','score','x_min','x_max','y_min','y_max'])

  # # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image)

  # process image
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  
  # # correct for image scale
  boxes /= scale
  
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < 0.3:
          break
      
      # Make coordinates as int type
      b = box.astype(int)

      df = df.append({'class_id':label,
                'score':score,
                'x_min':b[1],
                'x_max':b[3],
                'y_min':b[0],
                'y_max':b[2]}, ignore_index=True)


  # Calculate best bounding box coordinates for each class ={0,1}

  # Class 0 - surname
  class_0_best_score = df[df.class_id == 0].score.max()
  class_0_x_min = int(df[(df.class_id == 0)&(df.score == class_0_best_score)].x_min)
  class_0_y_min = int(df[(df.class_id == 0)&(df.score == class_0_best_score)].y_min)
  class_0_x_max = int(df[(df.class_id == 0)&(df.score == class_0_best_score)].x_max)
  class_0_y_max = int(df[(df.class_id == 0)&(df.score == class_0_best_score)].y_max)

  # Class 1 - id
  class_1_best_score = df[df.class_id == 1].score.max()
  class_1_x_min = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].x_min)
  class_1_y_min = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].y_min)
  class_1_x_max = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].x_max)
  class_1_y_max = int(df[(df.class_id == 1)&(df.score == class_1_best_score)].y_max)

  crop_surname = crop[class_0_x_min:class_0_x_max, class_0_y_min:class_0_y_max]
  crop_id = crop[class_1_x_min:class_1_x_max, class_1_y_min:class_1_y_max]
  
  return crop_surname, crop_id

# Rotation functions
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    #print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))

# Text from product_name
def text_from_img(image):
  img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  test_list = pytesseract.image_to_string(img_rgb).split('\n')

  
  # using join() + generator to remove special characters
  special_char = '@_!#$^&*()<>?/\|}{~:;.[]'
  out_list = [' '.join(test_list)]

  # trim spaces
  out_list = [elem.strip() for elem in out_list]

  # remove Null elements
  #out_list = list(filter(None, out_list))

  return out_list[0]

def detect_text(img_path, model):
  image_loaded = read_image_bgr(img_path)

  croped_surname, croped_id = model_elements(image_loaded, model)

  surname_deskewed, id_deskewed = deskew(croped_surname), deskew(croped_id)

  text_surname = text_from_img(surname_deskewed)

  text_id = text_from_img(id_deskewed)

  return text_surname, text_id


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        res = detect_text(file_path, model)
        
        # Convert result to string
        result = 'Surname :' + res[0] +', id : ' + res[1]
        
        #remove saved file
        os.remove(file_path)
        return result
    return None


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

