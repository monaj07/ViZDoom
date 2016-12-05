
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, Dropout
from keras.layers import Flatten
from keras.models import Sequential

def vggModel(weights_path=None, DROP_RATIO=0.5):
    CNN_model = Sequential()
    CNN_model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    CNN_model.add(Convolution2D(64, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(64, 3, 3, activation='relu'))
    CNN_model.add(MaxPooling2D((2,2), strides=(2,2)))

    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(128, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(128, 3, 3, activation='relu'))
    CNN_model.add(MaxPooling2D((2,2), strides=(2,2)))

    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(256, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(256, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(256, 3, 3, activation='relu'))
    CNN_model.add(MaxPooling2D((2,2), strides=(2,2)))

    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(512, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(512, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(512, 3, 3, activation='relu'))
    CNN_model.add(MaxPooling2D((2,2), strides=(2,2)))

    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(512, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(512, 3, 3, activation='relu'))
    CNN_model.add(ZeroPadding2D((1,1)))
    CNN_model.add(Convolution2D(512, 3, 3, activation='relu'))
    CNN_model.add(MaxPooling2D((2,2), strides=(2,2)))

    CNN_model.add(Flatten())
    CNN_model.add(Dense(4096, activation='relu'))
    CNN_model.add(Dropout(DROP_RATIO))
    CNN_model.add(Dense(4096, activation='relu'))
    CNN_model.add(Dropout(DROP_RATIO))
    CNN_model.add(Dense(1000, activation='softmax'))

    if weights_path is not None:
        CNN_model.load_weights(weights_path)
    return CNN_model
