import os
from collections import namedtuple
import numpy
from keras import backend as keras_backend
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils


def _init_modules():
    """Initialize needed modules."""
    # use Theano image dimensions notation
    keras_backend.set_image_dim_ordering('th')
    # fix seed value to ensure reproducibility
    numpy.random.seed(42)


_init_modules()


TrainData = namedtuple('TrainData', ['x_train', 'y_train', 'x_test', 'y_test', 'result_categories'])


class DigitRecognizer:
    """Class containing logic for MNIST digit recognition via coonvolutional neural network."""
    def __init__(self, weights_file_path='data/cnn_mnist_weights.hdf5'):
        self.model = self._create_recognizer_model(weights_file_path)

    def _create_cnn_model(self, train_data):
        """Creates non-trained convolutional neural network model for recognizing MNIST digits from 28×28 images."""
        cnn_model = Sequential()
        # convolutional layer with 30 feature maps of size 5×5
        cnn_model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        # pooling layer taking the max over 2×2 patches
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        # convolutional layer with 15 feature maps of size 3×3
        cnn_model.add(Conv2D(15, (3, 3), activation='relu'))
        # pooling layer taking the max over 2×2 patches
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        # randomly drop 20% of neurons in order to reduce over fitting to limited sample data
        cnn_model.add(Dropout(0.2))
        # reduce number of dimensions since only classification output is needed
        cnn_model.add(Flatten())
        # fully connected layer with 128 neurons and rectifier activation - to connect all the data
        cnn_model.add(Dense(128, activation='relu'))
        # fully connected layer with 50 neurons and rectifier activation
        cnn_model.add(Dense(50, activation='relu'))
        # squash data into output probabilities
        cnn_model.add(Dense(train_data.result_categories, activation='softmax'))

        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('CNN model is created.')
        return cnn_model

    def _train_cnn_model(self, cnn_model, train_data, weights_file_path):
        """Trains convolutional neural network and save resulting weights to file."""
        cnn_model.fit(train_data.x_train, train_data.y_train,
                      validation_data=(train_data.x_test, train_data.y_test),
                      epochs=10,
                      batch_size=200)
        scores = cnn_model.evaluate(train_data.x_test, train_data.y_test, verbose=0)
        error = 100 - scores[1] * 100
        print(f'CNN MNIST data set digit recognition error: {error}%.')
        cnn_model.save_weights(weights_file_path, overwrite=True)
        print('CNN model is trained and weights are saved for future use.')

    def _get_mnist_train_data(self):
        """Prepares MNIST data set train data."""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = self._reshape_sample_data(x_train)
        x_test = self._reshape_sample_data(x_test)

        x_train = self._normalize_sample_data(x_train)
        x_test = self._normalize_sample_data(x_test)

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]

        return TrainData(x_train=x_train,
                         y_train=y_train,
                         x_test=x_test,
                         y_test=y_test,
                         result_categories=num_classes)

    def _reshape_sample_data(self, sample_data):
        """Reshapes sample data to following format: [samples][pixels][width][height]."""
        return sample_data.reshape(sample_data.shape[0], 1, 28, 28).astype('float32')

    def _normalize_sample_data(self, sample_data):
        """Normalizes inputs from 0-255 to 0-1."""
        return sample_data / 255

    def _create_recognizer_model(self, weights_file_path):
        """Creates MNIST digit recognizer model."""
        train_data = self._get_mnist_train_data()
        cnn_model = self._create_cnn_model(train_data)

        if os.path.isfile(weights_file_path):
            cnn_model.load_weights(weights_file_path)
            print('CNN weights are loaded.')
        else:
            self._train_cnn_model(cnn_model, train_data, weights_file_path)

        print('Digit recognizer model is created.')
        return cnn_model

    def recognize_digit(self, input_shape):
        """Recognizes digit represented by provided shape."""
        result = self.model.predict(input_shape)
        result = result.reshape(1, 10)
        digit = int(numpy.argmax(result))
        return digit
