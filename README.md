# About dog_breed_app
This is a flask app that displays a dog breed based off a classification algorithm I wrote using Keras.

## Overview
This codebase utilizes a few datasets of labeled dog images in order to train a Convolutional Neural Network (CNN).

### Important successes:
1. Build a pipeline that takes in user provided images and returns the predicted breed associated with that image.
2. Intrinsic to this pipeline is a Sequential CNN model built using Keras.
3. The accuracy of the algorithm is over 80% accurate, which is pretty great considering how hard it is for humans to tell the breeds of dogs sometimes!

### Notes
In order to run this package you will need to configure a few things!
1. I reconfigured most of the code to rerun my algorithm if I desired with a different/larger dataset. This still needs some tweaking, because of a lot of the paths, but it could be ready to go shortly
2. I defaulted this model to run on Resnet50, but the code should be default enough that we can easily swap this over to VGG19 with a little bit of tweaking.
3. I fooled around on being able to take an image URL instead of a local file. This isn't complete yet!

### Needed python packages
1. Flask - flexible and easy to code up web app (We use werkzeug as well...)
2. Keras - for Machine learning neural net stuff
3. Pandas - for data manipulation
4. Numpy - for data manipulation
5. Plotly - for pretty graphs in the flask app
6. CV2 - from Open CV

### Running this locally
1. By downloading the full package as listed you should be able to clone it.
2. Open a Terminal or Commandline window and get to the app directory
3. Run 'python run.py' in the command line and it should render at '127.0.0.1:5000'
