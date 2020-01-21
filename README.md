# About dog_breed_app
This is a flask app that displays a dog breed based off a classification algorithm I wrote using Keras.

## Overview
This codebase utilizes a few datasets of labeled dog images in order to train a Convolutional Neural Network (CNN).

### Important successes:
1. Build a pipeline that takes in user provided images and returns the predicted breed associated with that image.
2. Intrinsic to this pipeline is a Sequential CNN model built using Keras.
3. The accuracy of the algorithm is 80% accurate on the test set, which is pretty great considering how hard it is for humans to tell the breeds of dogs sometimes!

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

### Metrics
If you would like to know more about the metrics associated with this project please refer to the jupyternotebook in the outerdirectory.
Metrics of note:
1. Accuracy of Algorithm on Test Set: 79.30% (in my last run)
2. About the training set:
-There are 6680 training dog images.
-There are 835 validation dog images.
-There are 836 test dog images.
3. Number of Epochs Run: 20
4. Best epoch was from Epoch 3: This is the model that we import to predict breeds. On that saved model these are the following metrics: loss: 0.3491 - acc: 0.8920, val_loss improved from 0.70242 to 0.64562

## Methodology
### Preprocessing the Data and Initial Data Investigation
For preprocessing the data I followed this path:
1. Import the data by using sklearn datasets load_files method. The dataset was provided in numpy arrays so using this library is helpful.
2. We then converted the target variables (aka our dog breeds) into an array using the to_categorical method within keras.utils
3. Our dataset was already split for us between valid, test and training data, so we pulled those datasets in as necessary. In the future, I think it would be best to rearrange these datasets so that I could make sure an avoid overfitting.
4. We pulled out the dog names associated from the data (133 in all).
5. I also pulled in pictures of humans for testing my human algorithm.

Development of face detector and dog detector:
1. There is a wonderful library that Udacity provided us called haarcascades this is used for detecting faces.
2. I then tested this against our datasets

### Implementation
Document machine learning pipeline and why I chose the one I did
I ended up using a relatively simple Keras Resnet50 pipeline. I initially trained my algorithm using the VGG19 dataset but it overfit to the 

### Process Refinement
I tried the following four packages in the Udacity provided workplace that is connected to a much faster GPU:
1. locally to test out the accuracy given the simple model I wanted to use.
I found that the Resnet50 algorithm was closest to what I wanted but I also could have spent more time fine tuning my pipeline then I had time for. Going forw


