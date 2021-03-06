import json
import plotly
import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template, request
import cv2                  
from tqdm import tqdm
import sys
#from dog_breed_app.models.load_dog_data import *
parent_folder = sys.path[5].split("\\app")[0]
sys.path.append(parent_folder)
from models.run_model import *
#from models.load_dog_data import *
from data.dog_names import *
from helpers import *
import os
from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras import backend as K
# load model
algo_type = 'Resnet50'
model = get_model(algo_type="Resnet50")

def predict_breed(img_path, model):
    # extract bottleneck features
    if algo_type == 'Resnet50':
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    elif algo_type == 'VGG19':
        bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def get_dog_info(img_path, model):
    '''
    Input image path and CNN model

    Output string about the said image using the models
    '''
    dog_name = predict_breed(img_path, model).split('.')[1]
    if dog_detector(img_path, model) and face_detector(img_path):
        title = "We can't tell if this is a dog or a human, but either way it looks like a {}".format(dog_name)
    elif dog_detector(img_path, model):
        title = "This is a dog and it looks like a/an {} dog breed".format(dog_name)
    elif face_detector(img_path):
        title = "This is a human and it looks like a/an {} dog breed".format(dog_name)
    else:
        title = "We couldn't seem to find humans or dogs here but it looks like a {}".format(dog_name)
    return title


app = Flask(__name__)

UPLOAD_FOLDER = "static/images/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit(".",1)[1].lower() in  ALLOWED_EXTENSIONS

default_image_path = os.path.join(UPLOAD_FOLDER,os.listdir(UPLOAD_FOLDER)[0])
#accuracy = 
print(default_image_path)
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    K.clear_session()
    model = get_model(algo_type="Resnet50")
    default_image = default_image_path
    message = get_dog_info(default_image, model)
    K.clear_session()
    # render web page with plotly graphs
    return render_template('master.html', image_file=default_image, message=message)

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        image_file = default_image_path
    if request.method == 'POST':
        f = request.files['file']
        saved_filename = UPLOAD_FOLDER+secure_filename(f.filename)
        f.save(saved_filename)
        print(f.filename)
        image_file = saved_filename
    K.clear_session()
    model = get_model(algo_type="Resnet50")
    message = get_dog_info(image_file, model).replace("_", " ")
    K.clear_session()
    # render web page with plotly graphs
    return render_template('master.html', image_file=image_file, message=message)

# web page that handles user query and displays model results
@app.route('/image_link', methods=['GET', 'POST'])
def image_link():
    # save user input in query
    if request.method == 'GET':
        image_file = default_image_path
    if request.method == 'POST':
        image_file = request.args.get('image_link') 
    K.clear_session()
    model = get_model(algo_type="Resnet50")
    message = get_dog_info(image_file, model)
    K.clear_session()
    return render_template(
        'go.html',
        image_file=image,
        message=message
    )

def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()