from tensorflow import keras
import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

# Neural Network Shape:
# 1 input layer, 2 hidden layers, one output layer

# LOAD THE DATA

# With mnist.load_data() we get DATA and LABELS for testing and training sets
# We split training data & test data set with a 25% split
# Most of the data is TRAINING data
# Test data is a percentage% of the total data needed to EVALUATE the model

# y data is the labels (0 to 9), this is why we don't scale down the y dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_shape = x_train.shape
y_shape = y_train.shape
print(x_shape, y_shape)


# NORMALIZE & PREPROCESS THE DATA

# We scale down the data to (0,1) to get the ratios and compute it easier

# The CNN requires one more MOCK dimension (We cannot feed the NN not formatted input)
# We need this MOCK dimension to reshape the matrix of inputs

# DATA is ( [dim train data], resolution_x, resolution_y ) - shaped
# FORMATTED DATA IS ( [dim train data], resolution_x, resolution_y, 1 ) - shaped


# 1) if AXIS = NONE, input is considered a vector => one vector norm for all values in the tensor
# ....... => norm(reshape(tensor, [-1]))
# 2) if AXIS = INTEGER, input is considered a batch of vectors, and axis determines the axis in
# ....... tensor over which to compute vector norms
# 3) if AXIS = TUPLE, input is a matrix, same approach as 2)

# Example: If you are passing a tensor that can be either a matrix or a batch of matrices at runtime, pass axis=[-2,-1]


def train_without_convolution(train_data_x, test_data_x, train_data_y, test_data_y):
    x_train_nn = tf.keras.utils.normalize(train_data_x, axis=1)
    x_test_nn = tf.keras.utils.normalize(test_data_x, axis=1)
    y_train_nn = train_data_y
    y_test_nn = test_data_y

    num_classes = 10

    y_train_nn = keras.utils.to_categorical(y_train_nn, num_classes)
    y_test_nn = keras.utils.to_categorical(y_test_nn, num_classes)

    print('x_train_nn shape:', x_train_nn.shape)
    print(x_train_nn.shape[0], 'train samples')
    print(x_test_nn.shape[0], 'test samples')

    batch_size = 1
    epochs = 6

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax))
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    hist = model.fit(x_train_nn, y_train_nn, batch_size=batch_size, epochs=epochs,
                     validation_data=(x_test_nn, y_test_nn))
    print("The model has successfully trained")

    loss, accuracy = model.evaluate(x_test_nn, y_test_nn)
    print(accuracy)
    print(loss)
    model.save('digits.model')

    for x in range(0, 10):
        img = cv.imread(f'{x}.png')[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("with a prediction of " + str(accuracy))
        print(f' the result is: {np.argmax(prediction)}')

        # plt.imshow(img[0], cmap=plt.cm.binary)
        plt.imshow(img[0])
        plt.show()


def train_with_convolution(train_data_x, test_data_x, train_data_y, test_data_y):
    # x_train_nn = tf.keras.utils.normalize(train_data_x, axis=-1)
    # x_test_nn = tf.keras.utils.normalize(test_data_x, axis=-1)
    x_train_nn = train_data_x
    x_test_nn = test_data_x
    y_train_nn = train_data_y
    y_test_nn = test_data_y
    # basic forwarding neural network

    x_train_nn = x_train_nn.reshape(x_train_nn.shape[0], 28, 28, 1)
    x_test_nn = x_test_nn.reshape(x_test_nn.shape[0], 28, 28, 1)

    # y_train_nn = y_train_nn.reshape(y_train_nn.shape[0], 28, 28, 1)
    # y_test_nn = y_test_nn.reshape(y_test_nn.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)
    num_classes = 10

    # convert class vectors to binary class matrices
    y_train_nn = keras.utils.to_categorical(y_train_nn, num_classes)
    y_test_nn = keras.utils.to_categorical(y_test_nn, num_classes)

    x_train_nn = x_train_nn.astype('float32')
    x_test_nn = x_test_nn.astype('float32')
    x_test_nn /= 256
    x_train_nn /= 256

    print('x_train shape:', x_train_nn.shape)
    print(x_train_nn.shape[0], 'train samples')
    print(x_test_nn.shape[0], 'test samples')

    # 3. Create the model
    # Now we will create our CNN model in Python data science project.
    # A CNN model generally consists of convolutional and pooling layers.
    # It works better for data that are represented as grid structures,
    # this is the reason why CNN works well for image classification problems.
    # e will then compile the model with the Adadelta optimizer.

    batch_size = 128
    epochs = 10

    # basic sequential in-order NN (simple linear NN)
    model = tf.keras.models.Sequential()

    # x_train_nn = x_train_nn.reshape(-1, 28, 28, 1)
    # x_test_nn = x_test_nn.reshape(-1, 28, 28, 1)

    # flattening layer for input
    # 280x280 ing -> transformed into a single one dim-array (for every img individually)
    # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # first args = inputs (number of neurons)
    # second args = kernel_size
    # model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # pooling layer
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # adding dropout layer to prevent ovefitting
    # by fitting all the weights with a 1/(1 - rate)

    model.add(Dense(10, activation='relu'))

    # output layer
    # units = nr of classes = nr of digits
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train_nn, y_train_nn, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_test_nn, y_test_nn))
    print("The model has successfully trained")
    # 4. Train the data
    # The model.fit() function of Keras will start the training of the model.
    # It takes the training data, validation data, epochs, and batch size.
    # It takes some time to train the model. After training, we save the weights and model definition in the ‘mnist.h5’ file

    print("The model has successfully trained")

    # # 5. Evaluate the model
    # # We have 10,000 images in our dataset which will be used to evaluate how good our model works.
    # # The testing data was not involved in the training of the data therefore, it is new data for our model.
    # # The MNIST dataset is well balanced so we can get around 99% accuracy.
    loss, accuracy = model.evaluate(x_train, y_train)
    print(accuracy)
    print(loss)

    # saved the weights and etc. in the digits.model directory
    model.save('digits.model')

    # read images with open cv

    for x in range(1, 10):
        # all of it and the first of the last one
        img = cv.imread(f'{x}.png')
        img = img.resize((28, 28))
        img = np.array([img])
        #img = img.reshape(1, 28, 28, 1)
        img = img / 256

        # the softmax results of all the output neurons
        # low likely it is to be a 0 ... 9

        # the index of the neuron is equivalent ot the class
        # index 6 => result is 6
        prediction = model.predict([img])[0]

        # get the index of the highest value
        # it prints the most certain predicted case

        print("with a prediction of " + str(prediction))
        print(f' the result is: {np.argmax(prediction)}')

        # plt.imshow(img[0], cmap=plt.cm.binary)
        plt.imshow(img[0])
        plt.show()

        saved_model = tf.keras.models.load_model(model)


def start():
    var = input(" > Press 1 for a convoluted NN / 0 for a non-convoluted NN ... ")
    if var == '1':
        train_with_convolution(x_train, x_test, y_train, y_test)
    elif var == '0':
        train_without_convolution(x_train, x_test, y_train, y_test)


start()
