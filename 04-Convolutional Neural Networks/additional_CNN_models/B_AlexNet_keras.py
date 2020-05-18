from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
np.random.seed(42)
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split

def create_model(img_shape):
    model = Sequential()

    # 1st Convolutional Layer and pooling
    model.add(Conv2D(filters=96, input_shape=img_shape, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # 2nd Convolutional Layer and pooling
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # 3rd Convolutional Layer *3 then max Pool
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())

    # Passing it to fully connected  layer
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(units=17, activation='softmax'))
    return model

def main():
    x, y = oxflower17.load_data(one_hot=True)
    AlexNet = create_model(img_shape=(224,224,3))
    print(AlexNet.summary())

    AlexNet.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    AlexNet.fit(x, y, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)
if __name__=='__main__':
    main()