from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from PIL import ImageFile

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('../data/dog_images/train')
valid_files, valid_targets = load_dataset('../data/dog_images/valid')
test_files, test_targets = load_dataset('../data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../data/dog_images/train/*/"))]
                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
