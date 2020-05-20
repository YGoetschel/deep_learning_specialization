import keras
import tensorflow as tf
import keras.backend as K
from keras.datasets import cifar10
import cv2
import numpy as np
from keras.utils import np_utils


def load_cifar10_data(img_rows, img_cols):
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    print(X_train.shape)
    num_classes = len(np.unique(Y_train))

    # resize image
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:, :, :, :]])
    X_valid = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_valid[:, :, :, :]])
    # normalize
    X_train = X_train/255.
    X_valid = X_valid/255.

    Y_train = np_utils.to_categorical(y=Y_train, num_classes=num_classes, dtype='float32')
    Y_valid = np_utils.to_categorical(y=Y_valid, num_classes=num_classes, dtype='float32')
    return X_train, Y_train, X_valid, Y_valid, num_classes

"""
In Keras the default kernel initializers are:
kernel_initializer="glorot_uniform"
bias_initializer="zeros"
"""

def inception_module(x, nb_filters, name=None):
    # # nb filters=64, f=1 s=1, p=same
    conv_1x1 = keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(1,1), strides=(1,1), padding='same',
                                   activation='relu')(x)

    # nb filters=96, f=1 s=1, p=same --> then --> nb filters=128, f=3 s=1, p=same
    conv_3x3a = keras.layers.Conv2D(filters=nb_filters[1][0], kernel_size=(1,1), strides=(1,1), padding='same',
                                    activation='relu')(x)
    conv_3x3b = keras.layers.Conv2D(filters=nb_filters[1][1], kernel_size=(3,3), strides=(1,1), padding='same',
                                    activation='relu')(conv_3x3a)

    # nb filters=16, f=1 s=1, p=same --> then --> nb filters=32, f=5 s=1, p=same
    conv_5x5a = keras.layers.Conv2D(filters=nb_filters[2][0], kernel_size=(1,1), strides=(1,1), padding='same',
                                    activation='relu')(x)
    conv_5x5b = keras.layers.Conv2D(filters=nb_filters[2][1], kernel_size=(5, 5), strides=(1, 1), padding='same',
                                    activation='relu')(conv_5x5a)

    # Max pool f=3, s=1, p=same  --> then  --> nb_filters=32, f=1, s=1, p=same
    max_pool = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1, 1), padding='same')(x)
    max_pool_conv = keras.layers.Conv2D(filters=nb_filters[3], kernel_size=(1,1), strides=(1,1), padding='same',
                                        activation='relu')(max_pool)

    output = keras.layers.Concatenate([conv_1x1, conv_3x3b, conv_5x5b, max_pool_conv], axis=3, name=name)  # axis=3 as contenate along channels
    return output


def auxiliary(x, num_classes, name=None):
    layer = keras.layers.AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='valid')(x)
    layer = keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(layer)
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Dense(units=256, activation='relu')(layer)
    layer = keras.layers.Dropout(0.4)(layer)
    layer = keras.layers.Dense(units=num_classes, activation='softmax', name=name)(layer)
    return layer


def nn_architecture(image_shape, num_classes):
    input_layer = keras.layers.Input(shape=image_shape)

    """Stage 1"""
    # output size of above operation: (224,224,3)
    x = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu',
                            name='conv_1_f=7_s=2')(input_layer)
    # output size of above operation: (112,112,64)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='MaxPool_1_f=3_s=2')(Z_1)
    # output size of above operation: (56,56,64)
    x = keras.layers.BatchNormalization()(x)

    """Stage 2"""
    x = keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',
                            name='conv_2_f=3_s=1')(x)
    # output size of above operation: (56,56,192)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_2_f=3_s=2')(x)
    # output size of above operation: (28,28,192)

    """Stage 3"""
    x = inception_module(x=x, nb_filters=[64, (96, 128), (16, 32), 32], name='inception_3a')
    # output size of above operation: (28,28,256)
    x = inception_module(x=x, nb_filters=[128, (128, 192), (32, 96), 64], name='inception_3b')
    # output size of above operation: (28,28,480)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='MaxPool_3_f=3_s=2')(x)
    # output size of above operation: (14,14,480)

    """Stage 4"""
    x = inception_module(x=x, nb_filters=[192, (96,208), (16,48), 64], name='inception_4a')
    # output size of above operation: (14,14,512)
    aux1 = auxiliary(x=x, num_classes=num_classes, name='aux1')
    x = inception_module(x=x, nb_filters=[160, (112,224),  (24,64),  64], name='inception_4b')
    # output size of above operation: (14,14,512)
    x = inception_module(x=x, nb_filters=[128, (128,256),  (24,64),  64], name='inception_4c')
    # output size of above operation: (14,14,512)
    x = inception_module(x=x, nb_filters=[112, (144, 288), (32, 64), 64], name='inception_4d')
    # output size of above operation: (14,14,528)
    aux2 = auxiliary(x=x, num_classes=num_classes, name='aux2')
    x = inception_module(x=x, nb_filters=[256, (160,320), (32,128), 128], name='inception_4e')
    # output size of above operation: (14,14,832)
    x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')

    """Stage 5"""
    x = inception_module(x=x, nb_filters=[256, (160,320), (32,128), 128], name='inception_5a')
    x = inception_module(x=x, nb_filters=[384, (192, 384), (48, 128), 128], name='inception_5b')
    x = keras.layers.AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='valid', name='AvgPool_5_f=7_s=1')(x)

    """Stage 6"""
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(rate=0.4)(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.Dense(10, activation='softmax', name='output')(x)

    model = keras.models.Model(inputs=input_layer, output=[x, aux1, aux2], name='inception_v1')
    return model


def main():
    X_train, Y_train, X_valid, Y_valid, num_classes = load_cifar10_data(224, 224)
    googLeNet = nn_architecture(image_shape=(224,224,3), num_classes=num_classes)
    print(googLeNet.summary())

    breakpoint()
    googLeNet.compile(optimizer='Adam',
                      loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                      metrics=['accuracy'])

if __name__ == '__main__':
    main()

