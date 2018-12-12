#!/usr/bin/python
import numpy as np
from os import path, walk
import copy, gc, cPickle, cv2, sys

#tensorflow dependencies
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, max_pool_1d, conv_1d
from tflearn.layers.estimator import regression
from sklearn.utils import shuffle

# save data to a pickle (.pck) file
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(obj, fp)

# restore() function to restore the pickle file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = cPickle.load(fp)
    return obj

def build_tflearn_convnet(dims=(120,320)): 
    
    input_layer = input_data(shape=[None, dims[0], dims[1], 1])
    conv_layer = conv_2d(input_layer,
                      nb_filter=20,
                      filter_size=5,
                      activation='sigmoid',
                      name='conv_layer_1')

    pool_layer  = max_pool_2d(conv_layer, 4, name='pool_layer_1')

    conv_layer2 = conv_2d(pool_layer,
                      nb_filter=20,
                      filter_size=5,
                      activation='sigmoid',
                      name='conv_layer_2')

    pool_layer2  = max_pool_2d(conv_layer2, 4, name='pool_layer_2')

    conv_layer3 = conv_2d(pool_layer2,
                      nb_filter=20,
                      filter_size=5,
                      activation='sigmoid',
                      name='conv_layer_3')

    pool_layer3  = max_pool_2d(conv_layer3, 4, name='pool_layer_2')

    fc_layer_1  = fully_connected(pool_layer3, 100,
                              activation='sigmoid',
                              name='fc_layer_1')

    fc_layer_2 = fully_connected(fc_layer_1, 4,
                             activation='softmax',
                             name='fc_layer_2')

    net = regression(fc_layer_2, optimizer='sgd',
                     loss='categorical_crossentropy',
                     learning_rate=0.05)

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
            # print '--------------',dir_path+file_path
            loaded_files.append(loadImg(dir_path+file_path))
    return loaded_files

def massageForCNN(data, dims=(120,320)):#dims is the dimensions of the image (raw images are 120x320 px; processed images are 240x320 px)
    result = np.ndarray(shape=(len(data),dims[0],dims[1],1))
    for i, outer in enumerate(data):
        for j, inner in enumerate(outer):
            for k, item in enumerate(inner):
                result[i][j][k] = np.array(item)
    return result

def trainAndSaveCNN(fp,trainX,trainY,testX,testY,cnn_type='2d',dims=(120,320)):
    NUM_EPOCHS = 15
    BATCH_SIZE = 20
    if cnn_type == '2d':
        MODEL = build_tflearn_convnet(dims)
        MODEL.fit(trainX, trainY, n_epoch=NUM_EPOCHS, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=BATCH_SIZE, run_id='Img_ConvNet')
        MODEL.save(fp)#persist convNet

    return MODEL

def startTraining(data, testData=None, dataset='raw', data_in_memory=False):
    TRAIN_IMG_CNN = True #used for debugging
    DATA_IN_MEMORY = data_in_memory #if training data is already in memory, no need to reprocess it

    #full img dataset
    raw_fp = '../PI_CAR_DATA/rawImages/'
    proc_fp = '../PI_CAR_DATA/processedImages/'

    #partial img dataset
    # raw_fp = './PI_CAR_DATA/rawImages/'
    # proc_fp = './PI_CAR_DATA/processedImages/'

    if dataset == 'raw':
        print 'Training for Raw Images'
        print '#####################################'
        dims = (120,320)
        cnnFp = './cnns/raw_img_cnn.tfl'
    elif dataset == 'proc':
        print 'Training for Processed Images'
        print '#####################################'
        dims = (240,320)
        cnnFp = './cnns/proc_img_cnn.tfl'
    else:
        print 'Error: Invalid Dataset Type'
        sys.exit(1)

    if TRAIN_IMG_CNN:
        print '#### Image CNN ####'
        img_cnn_trainX_fp = './data/img_cnn_trainx.pck'
        img_cnn_trainY_fp = './data/img_cnn_trainy.pck'

        img_cnn_testX_fp = './data/img_cnn_testx.pck'
        img_cnn_testY_fp = './data/img_cnn_testy.pck'
        
        img_cnn_validX_fp = './data/img_cnn_validx.pck'
        img_cnn_validY_fp = './data/img_cnn_validy.pck'

        # img_cnn_trainX_fp = './data/proc_img_cnn_trainx.pck'
        # img_cnn_trainY_fp = './data/proc_img_cnn_trainy.pck'

        # img_cnn_testX_fp = './data/proc_img_cnn_testx.pck'
        # img_cnn_testY_fp = './data/proc_img_cnn_testy.pck'
        
        # img_cnn_validX_fp = './data/proc_img_cnn_validx.pck'
        # img_cnn_validY_fp = './data/proc_img_cnn_validy.pck'
        
        if not DATA_IN_MEMORY:
            image_names = [item['Image File'] for item in data]

            print '#####################################'
            # load img training data
            if dataset == 'raw':
                raw_img_fp_arr = [img_name for img_name in image_names]
                imgs = loadFilesList(raw_fp, raw_img_fp_arr)
            elif dataset == 'proc':
                proc_img_fp_arr = [img_name for img_name in image_names]
                imgs = loadFilesList(raw_fp, proc_img_fp_arr)
            else:
                print 'Error: Invalid Dataset Type'
                sys.exit(1)
            print 'Image Data Loaded'

            print 'Starting to massage CNN data...'
            # massage training data for convNet
            #train x is input
            raw_img_X = massageForCNN(imgs,dims)
            
            # dictionary for converting commands to binary array ['up','right','down','left']
            COMMANDS_TO_ARR = {
                'up'    :[1,0,0,0],
                'right' :[0,1,0,0],
                'down'  :[0,0,1,0],
                'left'  :[0,0,0,1],
                ''      :[0,0,0,0]
            }

            #train y is expected output for trainX
            raw_img_Y = [[]]*len(imgs)
            for i,item in enumerate(data):
                raw_img_Y[i] = COMMANDS_TO_ARR[item['Commands']]
            print 'Data Massaged.'
            print raw_img_Y[:3] #print a few to make sure worked correctly

            print 'Cleaning up memory (data)...'
            del data
            gc.collect()

            print 'Establishing training/validation datasets (input)...'
            # split data for training sets to create input for ConvNet
            img_cnn_trainX = raw_img_X[0:(len(raw_img_X)*3/4)]
            img_cnn_validX = raw_img_X[(len(raw_img_X)*3/4):]

            print 'Cleaning up memory (raw_img_x)...'
            del raw_img_X
            gc.collect()

            print 'Establishing training/validation datasets (output)...'
            img_cnn_trainY = raw_img_Y[0:(len(raw_img_Y)*3/4)]
            img_cnn_validY = raw_img_Y[(len(raw_img_Y)*3/4):]

            print 'Cleaning up memory (raw_img_y)...'
            #explicitly clean up memory
            del raw_img_Y
            gc.collect()
            
            print 'Saving training data...'
            save(img_cnn_trainX,img_cnn_trainX_fp) #save training data
            save(img_cnn_trainY,img_cnn_trainY_fp)
            save(img_cnn_validX,img_cnn_validX_fp) #save validation data
            save(img_cnn_validY,img_cnn_validY_fp)
        else:
            print 'Cleaning up memory (data)...'
            #explicitly clean up memory
            del data
            gc.collect()

            print "Loading Data from Memory :)"
            img_cnn_trainX = load(img_cnn_trainX_fp)
            img_cnn_trainY = load(img_cnn_trainY_fp)
            img_cnn_validX = load(img_cnn_validX_fp)
            img_cnn_validY = load(img_cnn_validY_fp)
            print "Data Loaded."
###################################################################################################################
        # #massage Test Data data for convNet
        # b_img_testX = massageForCNN(bee_imgs_t)
        # b_img_testY = [[1, 0]]*len(bee_imgs_t)
        # nb_img_testX = massageForCNN(no_bee_imgs_t)
        # nb_img_testY = [[0, 1]]*len(no_bee_imgs_t)

        # img_cnn_testX = np.concatenate((b_img_testX,nb_img_testX))
        # img_cnn_testY = np.concatenate((b_img_testY,nb_img_testY))

        # save(img_cnn_testX,img_cnn_testX_fp) #save test data
        # save(img_cnn_testY,img_cnn_testY_fp)
        # print 'Done massaging CNN data'

        # #train convNet
        model = trainAndSaveCNN(cnnFp,img_cnn_trainX,img_cnn_trainY,img_cnn_validX,img_cnn_validY,dims=dims)
        print model.predict(img_cnn_validX[0].reshape(-1,dims[0],dims[1],1))
        # print img_cnn_validX
        # for img in img_cnn_validX:
        #     print model.predict(img.reshape(-1,120,320,1))
