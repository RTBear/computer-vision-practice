#!/usr/bin/python
import cv2
import numpy as np
import cPickle
from os import path, walk
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

def build_tflearn_convnet(dims=(320,120)): 
    
    input_layer = input_data(shape=[None, dims[0], dims[1], 1])
    conv_layer = conv_2d(input_layer,
                      nb_filter=20,
                      filter_size=5,
                      activation='sigmoid',
                      name='conv_layer_1')

    pool_layer  = max_pool_2d(conv_layer, 2, name='pool_layer_1')

    conv_layer2 = conv_2d(pool_layer,
                      nb_filter=20,
                      filter_size=5,
                      activation='sigmoid',
                      name='conv_layer_2')

    pool_layer2  = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1  = fully_connected(pool_layer2, 100,
                              activation='sigmoid',
                              name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 2,
                             activation='softmax',
                             name='fc_layer_2')

    net = regression(fc_layer_2, optimizer='sgd',
                     loss='categorical_crossentropy',
                     learning_rate=0.1)

    model = tflearn.DNN(net)
    return model

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

def loadFilesList(dir_path, file_paths, ftype='image'):
    loaded_files = []
    if ftype == 'image':
        #load each file in 'file_paths' as an image
        for file_path in file_paths:
            print '--------------',dir_path+file_path
            loaded_files.append(loadImg(dir_path+file_path))
    return loaded_files

def massageForCNN(data, dims=(320,120)):#dims is the dimensions of the image (raw images are 320x120 px; processed images are 320x240 px)
    print 'datalen:',len(data)
    result = np.ndarray(shape=(len(data),dims[0],dims[1],1))
    for i, outer in enumerate(data):
        for j, inner in enumerate(outer):
            for k, item in enumerate(inner):
                result[i][j][k] = np.array(item)
    return result

def trainAndSaveCNN(trainX,trainY,testX,testY,cnn_type='2d'):
    NUM_EPOCHS = 10
    BATCH_SIZE = 10
    if cnn_type == '2d':
        MODEL = build_tflearn_convnet()
        MODEL.fit(trainX, trainY, n_epoch=NUM_EPOCHS,
                shuffle=True,
                validation_set=(testX, testY),
                show_metric=True,
                batch_size=BATCH_SIZE,
                run_id='Img_ConvNet')
        MODEL.save('./cnns/b_img_cnn.tfl')#persist convNet

    return MODEL

def startTraining(data):
    IMAGE_TRAIN = True

    TRAIN_IMG_CNN = False

    if IMAGE_TRAIN:
        #full img dataset
        # raw_fp = '../PI_CAR_DATA/rawImages/'
        # proc_fp = '../PI_CAR_DATA/processedImages/'

        #partial img dataset
        raw_fp = './PI_CAR_DATA/rawImages/'
        proc_fp = './PI_CAR_DATA/processedImages/'

        image_names = [item['Image File'] for item in data]
        print image_names

        raw_img_fp_arr = [raw_fp+img_name for img_name in image_names]
        proc_img_fp_arr = [proc_fp+img_name for img_name in image_names]

        print '#####################################'

        print 'raw',raw_img_fp_arr[:3]
        print '---------------------'
        print 'proc',proc_img_fp_arr[:3]

        # # load bee image training data
        # bee_imgs = loadFilesList(dir_path, file_list)
        # no_bee_imgs = loadFilesList(dir_path, file_list)

        # # load bee image test data
        # bee_imgs_t = loadFilesDir(bee_test_fp, 'image')
        # no_bee_imgs_t = loadFilesDir(no_bee_test_fp, 'image')
        # print 'Image Data Loaded'

        if TRAIN_IMG_CNN:
            print '#### Image CNN ####'
            img_cnn_trainX_fp = './data/img_cnn_trainx.pck'
            img_cnn_trainY_fp = './data/img_cnn_trainy.pck'

            img_cnn_testX_fp = './data/img_cnn_testx.pck'
            img_cnn_testY_fp = './data/img_cnn_testy.pck'
            
            img_cnn_validX_fp = './data/img_cnn_validx.pck'
            img_cnn_validY_fp = './data/img_cnn_validy.pck'

            print 'Starting to massage CNN data...'
            # massage training data for convNet
            #train x is input
            b_img_X = massageForCNN(bee_imgs)
            #train y is expected output for trainX
            b_img_Y = [[1, 0]]*len(bee_imgs)

            nb_img_X = massageForCNN(no_bee_imgs)
            nb_img_Y = [[0, 1]]*len(bee_imgs)

            #combine the b and no_b training sets to create input for ConvNet
            img_cnn_X = np.concatenate((b_img_X,nb_img_X))
            img_cnn_trainX = img_cnn_X[0:(len(img_cnn_X)*3/4)]
            img_cnn_validX = img_cnn_X[(len(img_cnn_X)*3/4):]
            
            img_cnn_Y = np.concatenate((b_img_Y,nb_img_Y))
            img_cnn_trainY = img_cnn_Y[0:(len(img_cnn_Y)*3/4)]
            img_cnn_validY = img_cnn_Y[(len(img_cnn_Y)*3/4):]

            save(img_cnn_trainX,img_cnn_trainX_fp) #save training data
            save(img_cnn_trainY,img_cnn_trainY_fp)
            save(img_cnn_validX,img_cnn_validX_fp) #save validation data
            save(img_cnn_validY,img_cnn_validY_fp)

            #massage Test Data data for convNet
            b_img_testX = massageForCNN(bee_imgs_t)
            b_img_testY = [[1, 0]]*len(bee_imgs_t)
            nb_img_testX = massageForCNN(no_bee_imgs_t)
            nb_img_testY = [[0, 1]]*len(no_bee_imgs_t)

            img_cnn_testX = np.concatenate((b_img_testX,nb_img_testX))
            img_cnn_testY = np.concatenate((b_img_testY,nb_img_testY))

            save(img_cnn_testX,img_cnn_testX_fp) #save test data
            save(img_cnn_testY,img_cnn_testY_fp)
            print 'Done massaging CNN data'

            #train convNet
            model = trainAndSaveCNN(img_cnn_trainX,img_cnn_trainY,img_cnn_testX,img_cnn_testY)
            print model.predict(img_cnn_testX[0].reshape(-1,32,32,1))

if __name__ == '__main__':
    # startTraining()

    # file_list = ['image_2018_10_25_18:44:37.png','image_2018_10_25_18:44:38.png','image_2018_10_25_18:44:39.png']
    # dir_path = './PI_CAR_DATA/rawImages/'

    # print loadFilesList(dir_path, file_list)

    pass
