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
4. Best epoch was from Epoch 3: This is the model that we import to predict breeds. On that saved model these are the following metrics:    training set: loss: 0.0720 - acc: 0.9871 - precision_m: 0.9902 - f1_m: 0.9860 - recall_m: 0.9820
  validation_set: val_loss: 0.6168 - val_acc: 0.8240 - val_precision_m: 0.8512 - val_f1_m: 0.8196 - val_recall_m: 0.7916
5. Using these metrics we see how we're overfitting to our training set, so we would want to tweak the split of our training, test and validation set in order to avoid this overfitting.



#### Why accuracy
The reason why I chose accuracy in this instance is that it's the easiest for me to comprehend for the goal of this project. The set goal of this project was to improve accuracy above 60%, and that's exactly what I was able to do with a very simple algorithm. It's often best for business purposes to use a simple model and a simple metric. In this instance we accomplished both. Here are some visualizations of the the accuracy, precision, recall and f1 score of the final model.

### Metric improvement
I think the best way to improve these metrics is to hone in and improve the accuracy between two breeds. If this was a more comprehensive study, we might want to see where and how we were missing on the test and training set.

## Methodology
### Preprocessing the Data and Initial Data Investigation
#### For preprocessing the data I followed this path:
1. Import the data by using sklearn datasets load_files method. The dataset was provided in numpy arrays so using this library is helpful.
2. We then converted the target variables (aka our dog breeds) into an array using the to_categorical method within keras.utils
3. Our dataset was already split for us between valid, test and training data, so we pulled those datasets in as necessary. In the future, I think it would be best to rearrange these datasets so that I could make sure an avoid overfitting.
4. We pulled out the dog names associated from the data (133 in all).
5. I also pulled in pictures of humans for testing my human algorithm.

#### Development of face detector and dog detector:
1. There is a wonderful library that Udacity provided us called haarcascades this is used for detecting faces.
2. I then tested this against a few images for fun and to make sure that it operated properly!

#### path_to_tensor and paths_to_tensor functions
1. Load the image and crop the image using target_size parameter
2. Conver the image to a 3D tensor using "img_to_array" built in from keras.preprocessing.image library
3. Tack on an extra dimension using numpy's expand dims.
4. We then vertically stack the tensors for all the images if we're given a group of images in the paths_to_tensor function so we can get them later.
Reasoning: We need a 4D array for our CNN with the following dimensions: (batch_size, height, weight, depth)

### Implementation
#### Dog and Human Detectors:
This piece of the code is relatively simple as a result of two datasets which I was provided: a pretrained Resnet50 algorithm and the haarcascades dataset for face detection.

#### Model building
I ended up using a relatively simple Keras Sequential Resnet50 pipeline which is as follows:


    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_6 ( (None, 2048)              0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________

After running this model, I export the best results into a hd5 for me to use with the flask app later (no need to rerun everytime!).

#### Flask app implementation
I tried to keep this app as simple as possible, but I ended up dreaming of other functionality that I could eventually develop.
Based on the time constraints, I was only able to keep a relatively simple design that allows the user to upload an image and then the website renders the image and explains what the image is!
This took a lot of hacking to troubleshoot, but in general, I love how great flask is to put something together!


### Process Refinement
I tried the following four packages in the Udacity provided workplace that is connected to a much faster GPU:
1. Xception
2. Inception
3. VGG19
4. Resnet50

I found that the Resnet50 algorithm had the best accuracy to what I wanted but I also could have spent more time fine tuning my pipeline then I had time for. I think this makes sense given the large amount of training that this CNN has completed using imagenet.

I initially tried a more complicated algorithm with many layers and the performance was poor, topping out at 40 %. After a quick googling of the problem at hand, and attempting to find a good algorithm for this problem I came across this article: https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d.

After coming across their example of using MaxPooling2D I decided to try again the GlobalAveragingPool2D. This would drop our dataset to 2 dimensions from 4 before we used a Dropout and Dense function to provide the results.

## Reflection
I think that I have a lot still to learn in this space. I was only really able to dip my toe into the world of Computer Vision and I hope that I can improve a lot here. 

I think the interesting and complicated piece for me was actually implementing this project into an app. I think I have much that I wish I had the time to complete including better visualizations of my model accuracy. It was difficult for me to get the shapes correct between my layers, so I think I have more to read and improve on.

#### Possilbe Improvements
1. I wish that I had finished coding the image link download functionality.
2. I think we should also reshuffle our test and training images. It's clear to me that we're overfitting for the training set.


