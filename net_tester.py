import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import tensorflow as tf
import numpy as np
from train import massageForCNN, loadImg

def fit_image_convnet(convnet, image_path, dims=(120,320)):
    img = loadImg(image_path)
    prediction = convnet.predict(img.reshape([-1,dims[0],dims[1],1]))[0]
    print "unprocessed prediction:",prediction
    if prediction[0] > prediction[1] and prediction[0] > prediction[2] and prediction[0] > prediction[3]:
        return 'up pressed'#np.array([1,0,0,0])
    elif prediction[1] > prediction[0] and prediction[1] > prediction[2] and prediction[1] > prediction[3]:
        return 'right pressed'#np.array([0,1,0,0])
    elif prediction[2] > prediction[0] and prediction[2] > prediction[1] and prediction[2] > prediction[3]:
        return 'down pressed'#np.array([0,0,1,0])
    elif prediction[3] > prediction[0] and prediction[3] > prediction[1] and prediction[3] > prediction[2]:
        return 'left pressed'#np.array([0,0,0,1])
    else:
        return 'none'

def loadImgCNN(fp,dims=(120,320)):
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
    model.load(fp, weights_only=True)
    return model

if __name__ == '__main__':
    img_cnn = loadImgCNN('./cnns/raw_img_cnn.tfl')
    fit_b_cnn_img = fit_image_convnet(img_cnn,'./PI_CAR_DATA/rawImages/image_2018_10_25_18:45:49.png')#correct result is: 
    print fit_b_cnn_img
    