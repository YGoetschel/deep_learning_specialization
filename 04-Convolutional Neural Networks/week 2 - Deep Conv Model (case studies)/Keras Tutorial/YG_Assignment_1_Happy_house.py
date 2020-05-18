import numpy as np
import keras
# from keras.layers Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *


""" MODEL BUILDING IN KERAS  - Supervised Learning """
def model(input_shape):
    """
    During each step of forward propagation, we are just writing the latest value in the commputation into the same variable X
    :param input_shape is tuple of shape (n_h, n_w, n_c)
    """
    # define input placeholder
    X_input = keras.layers.Input(input_shape)                                                 # output shape (64, 64, 3)
    # pad the border of X-input with zeros
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)               # output shape (70, 70, 3) p=3 to get same convolution
    # CONV -> Batch Norm -> RELU Block applied to X
    X = keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)                     # output shape (64, 64, 32)
    X = keras.layers.BatchNormalization(axis=3, name='bn0')(X)                               # output shape (64, 64, 32)
    X = keras.layers.Activation('relu')(X)                                                   # output shape (64, 64, 32)
    # MAXPOOL
    X = keras.layers.MaxPooling2D((2, 2), name='max_pool')(X)                                # output shape (32, 32, 32)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = keras.layers.Flatten()(X)                                                            # output shape (32768, 1)
    X = keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='Happy_Model_1')
    return model


""" MODEL BUILDING IN KERAS  - Supervised Learning """
def model_v2(input_shape):
    # define input placeholder
    X_input = keras.layers.Input(input_shape)                                                 # output shape (64, 64, 3)
    # pad the border of X-input with zeros
    X = keras.layers.ZeroPadding2D((1, 1))(X_input)               # output shape (66, 66, 3) p=1 to get same convolution
    # CONV -> Batch Norm -> RELU Block applied to X
    X = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X)                     # output shape (64, 64, 32)
    X = keras.layers.BatchNormalization(axis=3, name='bn0')(X)                               # output shape (64, 64, 32)
    X = keras.layers.Activation('relu')(X)                                                   # output shape (64, 64, 32)
    # MAXPOOL
    X = keras.layers.MaxPooling2D((2, 2), name='max_pool')(X)                                # output shape (32, 32, 32)
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = keras.layers.Flatten()(X)                                                            # output shape (32768, 1)
    X = keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='Happy_Model_2')
    return model


def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    # Y values are of shape (n_c, m) must be transposed
    y_train = Y_train_orig.T
    y_test = Y_test_orig.T

    models = {'model 1': model(X_train.shape[1:]),
              'model 2': model_v2(X_train.shape[1:])}
    for model_name, model_fx in models.items():
        model_fx.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   # Happy Challenge is a binary classification problem.

    # if you run fit() again, the model will continue to train with the parameters it has already learnt instead of reinitializing them.
        model_fx.fit(X_train, y_train, epochs=20, batch_size=64)
        preds = model_fx.evaluate(X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

        print(model_fx.summary())
if __name__=="__main__":
    main()