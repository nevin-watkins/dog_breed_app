import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request
import cv2                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from .models.dog_train_data import *
from helpers import *
import os
from flask import Flask, render_template, request
from werkzeug import secure_filename

# load model
model = get_model(algo_type="Resnet50")

def predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def get_dog_info(img_path):
    dog_name = Resnet50_predict_breed(img_path).split('.')[1]
    if dog_detector(img_path) and face_detector(img_path):
        title = "We can't tell if this is a dog or a human, but either way it looks like a {}".format(dog_name)
    elif dog_detector(img_path):
        title = "This is a dog and it looks like a/an {} dog breed".format(dog_name)
    elif face_detector(img_path):
        title = "This is a human and it looks like a/an {} dog breed".format(dog_name)
    else:
        title = "We couldn't seem to find humans or dogs here but it looks like a {}".format(dog_name)
    return title

DOG_FOLDER = os.path('../data/images/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = DOG_FOLDER
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    default_image = os.listdir(DOG_FOLDER)[0]
    message = get_dog_info(default_image)
    # render web page with plotly graphs
    return render_template('master.html', image=default_image, message=message)

@app.route('/upload_image', methods=['GET', 'POST'])
def uploader():
    if request.method == 'GET':
        default_image = os.listdir(DOG_FOLDER)[0]
        return render_template('master.html', image=default_image, message=message)
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
    message = get_dog_info(f.filename)
    # render web page with plotly graphs
    return render_template('master.html')

# web page that handles user query and displays model results
@app.route('/image_url', methods=['GET', 'POST'])
def image_url():
    # save user input in query
    image_url = request.args.get('image_url') 
    message = "We'll figure this out later"
    return render_template(
        'go.html',
        image_file=image_file,
        message=message
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()