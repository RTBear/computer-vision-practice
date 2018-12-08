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

if __name__ == '__main__':
    IMAGE_TRAIN = True

    TRAIN_IMG_CNN = True

    TRAIN_AUD_CNN = True


    if IMAGE_TRAIN:
        #full img dataset
        bee_train_fp = './BEE2Set/bee_train'
        bee_test_fp = './BEE2Set/bee_test'
        no_bee_train_fp = './BEE2Set/no_bee_train'
        no_bee_test_fp = './BEE2Set/no_bee_test'

        #partial img dataset
        # bee_train_fp = '../BEE2Set/bee_train'
        # bee_test_fp = '../BEE2Set/bee_test'
        # no_bee_train_fp = '../BEE2Set/no_bee_train'
        # no_bee_test_fp = '../BEE2Set/no_bee_test'

        # load bee image training data
        bee_imgs = loadFilesDir(bee_train_fp, 'image')
        no_bee_imgs = loadFilesDir(no_bee_train_fp, 'image')

        # load bee image test data
        bee_imgs_t = loadFilesDir(bee_test_fp, 'image')
        no_bee_imgs_t = loadFilesDir(no_bee_test_fp, 'image')
        print 'Image Data Loaded'

        if TRAIN_IMG_ANN:
            print '#### Image ANN ####'
            ann_img_train_d_fp = './data/ann_img_train_d.pck'
            ann_img_valid_d_fp = './data/ann_img_valid_d.pck'
            ann_img_test_d_fp = './data/ann_img_test_d.pck'

            #massage data for use with ANN
            ann_bee_imgs, ann_no_bee_imgs = massageForANN(bee_imgs,no_bee_imgs)

            train_input = ann_bee_imgs + ann_no_bee_imgs

            train_target = np.concatenate((np.array([np.array([[1],[0]])]*len(bee_imgs)),np.array([np.array([[0],[1]])]*len(no_bee_imgs))))

            ann_img_d = zip(train_input,train_target) 
            ann_img_train_d = ann_img_d[0:(len(ann_img_d)*3/4)]
            ann_img_valid_d = ann_img_d[(len(ann_img_d)*3/4):]
            
            # save training data
            save(ann_img_train_d,ann_img_train_d_fp)
            save(ann_img_valid_d,ann_img_valid_d_fp)

            #massage validation data
            ann_bee_imgs_t, ann_no_bee_imgs_t = massageForANN(bee_imgs_t, no_bee_imgs_t)

            # massage validation data to work with ann
            test_input = ann_bee_imgs_t + ann_no_bee_imgs_t
            
            ann_img_test_d = zip(test_input,[0]*len(bee_imgs_t)+[1]*len(no_bee_imgs_t))
            # save validation data
            save(ann_img_test_d,ann_img_test_d_fp)

            print "Starting Training ANN..."
            img_ann_data = createTrainSaveANN('./ImageANN.pck',ann_img_train_d, ann_img_test_d, 1024, 2)
            print img_ann_data

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