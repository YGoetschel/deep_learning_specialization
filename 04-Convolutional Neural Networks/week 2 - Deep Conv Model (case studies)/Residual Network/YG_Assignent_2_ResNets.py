import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from resnets_utils import *

"""
when developing very deep Neural networks, a huge barrier to training them is vanishing gradients: 
very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow.

n ResNets, 
a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:
"""

def identity_block(X, f, filters, stage, block):

    """
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: nb of filters in conv layer
    :param stage: int to name the layer, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :return: X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    Architecture:
    X -> Conv -> Batch Norm -> Relu -> Conv -> Batch Norm -> Relu -> Conv -> Batch Norm -> shortcut -> Relu
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid',
                            name=conv_name_base + '2a', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)  # axis=idx of n_c value
    X = keras.layers.Activation('relu')(X)

    # Second component of main path
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same',
                            name=conv_name_base + '2b', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)

    # third component of main path
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid',
                            name=conv_name_base + '2c', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # step of main path ->  Add shortcut value to main path, and pass it through a RELU activation
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Architecture:
    X --> Conv2D / Batch Norm / Relu --> Conv2D / Batch Norm / Relu --> Conv / Batch Norm --> shortcut(conv2d, batch Norm) --> Relu
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X
    # component 1
    X = keras.layers.Conv2D(filters=F1,kernel_size=(1,1),strides=(s,s),padding='valid', name=conv_name_base + '2a',
                            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    # component 2
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name= conv_name_base + '2b',
                            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    # component 3
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c',
                            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid',name=conv_name_base+'1',
                                     kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = keras.layers.Input(input_shape)                                                   # output shape (64,64,3)
    # Zero-Padding
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)                                             # output shape (70,70,3)
    # Stage 1
    X = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1',
                            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)   # output shape (32,32,64)
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)                            # output shape (32,32,64)
    X = keras.layers.Activation('relu')(X)                                                     # output shape (32,32,64)
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)                                   # output shape (15,15,64)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)            # output shape (15,15,64)->(15,15,64)->(15,15,256)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')                                # output shape (15,15,64)->(15,15,64)->(15,15,256)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')                                # output shape (15,15,64)->(15,15,64)->(15,15,256)

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)          # output shape (8,8,128)->(8,8,128)->(8,8,512)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')                              # output shape (8,8,128)->(8,8,128)->(8,8,512)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')                              # output shape (8,8,128)->(8,8,128)->(8,8,512)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')                              # output shape (8,8,128)->(8,8,128)->(8,8,512)

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)         # output shape (4,4,256)->(4,4,256)->(4,4,1024)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')                             # output shape (4,4,256)->(4,4,256)->(4,4,1024)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')                             # output shape (4,4,256)->(4,4,256)->(4,4,1024)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')                             # output shape (4,4,256)->(4,4,256)->(4,4,1024)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')                             # output shape (4,4,256)->(4,4,256)->(4,4,1024)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')                             # output shape (4,4,256)->(4,4,256)->(4,4,1024)

    # Stage 5 (≈3 lines)
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)     # output shape (2,2,512)->(2,2,512)->(2,2,2048)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')                             # output shape (2,2,512)->(2,2,512)->(2,2,2048)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')                             # output shape (2,2,512)->(2,2,512)->(2,2,2048)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)                     # output shape (1,1,2048)

    # output layer
    X = keras.layers.Flatten()(X)                                                              # output shape (2048,1)
    X = keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes),            # output shape (6,1)
                           kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = keras.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = np.eye(6)[Y_train_orig.reshape(-1)]
    Y_test = np.eye(6)[Y_test_orig.reshape(-1)]

    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit(X_train, Y_train, epochs=2, batch_size=32)
    # preds = model.evaluate(X_test, Y_test)
    # print("Loss = " + str(preds[0]))
    # print("Test Accuracy = " + str(preds[1]))
    print(model.summary())

if __name__=='__main__':
    main()