#!/usr/bin/python
import cv2
import numpy as np
import cPickle
from os import path, walk
from assn4 import createTrainSaveANN
import copy

#tensorflow dependencies
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, max_pool_1d, conv_1d
from tflearn.layers.estimator import regression
from sklearn.utils import shuffle

#audio processing
from scipy.io import wavfile

# save data to a pickle (.pck) file
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(obj, fp)

# restore() function to restore the pickle file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = cPickle.load(fp)
    return obj

def loadImg(fp):
    img = cv2.imread(fp)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert from RGB to grayscale
    scaled_gray_img = gray_img/255.0 #divide by 255.0 to scale the color value between 0 and 1
    return scaled_gray_img 

def loadFilesDir(dir_fp, ftype='image'):
    file_paths = []
    loaded_files = []
    if ftype == 'image':
        for (dirpath, dirnames, filenames) in walk(dir_fp):
            file_paths.extend(path.join(dirpath,filename) for filename in filenames if filename.endswith('.png'))
        #load each file in 'file_paths' as an image
        for file_path in file_paths:
            loaded_files.append(loadImg(file_path))
    return loaded_files