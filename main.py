import cv2
import imghdr
import numpy as np
import pathlib
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf
from scipy import io as spio
import idx2numpy
from matplotlib import pyplot as plt
from typing import *
import time
import os
from os.path import join

# def words_extraction(image_file: str, out_size=28) -> List[Any]:
#     img = cv2.imread(image_file)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     # rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
#     img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
#
#     # Get contours
#     contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     output = img.copy()
#
#     words = []
#     for idx, contour in enumerate(contours):
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         words_crop = gray[y:y + h, x:x + w]
#         # print(letter_crop.shape)
#
#         # Resize letter canvas to square
#         size_max = max(w, h)
#         letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
#         if w > h:
#             # Enlarge image top-bottom
#             y_pos = size_max // 2 - h // 2
#             letter_square[y_pos:y_pos + h, 0:w] = words_crop
#         elif w < h:
#             # Enlarge image left-right
#             x_pos = size_max // 2 - w // 2
#             letter_square[0:h, x_pos:x_pos + w] = words_crop
#         else:
#             letter_square = words_crop
#
#     cv2.imshow("Output", output)
#     cv2.waitKey(0)
#
#     # Sort array in place by X-coordinate
#     words.sort(key=lambda coordinate: coordinate[0], reverse=False)
#
#     return words

emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def letters_extract(image_file: str, out_size=28) -> List[Any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda coordinate: coordinate[0], reverse=False)

    cv2.imshow("Output", output)
    cv2.waitKey(0)

    return letters


def emnist_model():
    model = Sequential()
    model.add(
        Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def emnist_model2():
    model = Sequential()
    # In Keras there are two options for padding: same or valid. Same means we pad with the number on the edge and valid means no padding.
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    ## model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def emnist_model3():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model


def fit_model(model=None):
    t_start = time.time()

    global emnist_labels

    emnist_path = join('data', 'mnist')
    X_train = idx2numpy.convert_from_file(join(emnist_path, 'emnist-byclass-train-images-idx3-ubyte'))
    y_train = idx2numpy.convert_from_file(join(emnist_path, 'emnist-byclass-train-labels-idx1-ubyte'))

    X_test = idx2numpy.convert_from_file(join(emnist_path, 'emnist-byclass-test-images-idx3-ubyte'))
    y_test = idx2numpy.convert_from_file(join(emnist_path, 'emnist-byclass-test-labels-idx1-ubyte'))

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

    k = 10
    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    # Normalize
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    # Set a learning rate reduction
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5,
                                                                min_lr=0.00001)

    # Required for learning_rate_reduction:
    keras.backend.get_session().run(tf.global_variables_initializer())

    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
              batch_size=64, epochs=30,)
    print("Training done, dT:", time.time() - t_start)


def emnist_predict(model, image_file):
    img = keras.preprocessing.image.load_img(image_file, target_size=(28, 28), color_mode='grayscale')
    emnist_predict_img(model, img)


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict_classes([img_arr])
    return chr(emnist_labels[result[0]])


def img_to_str(model: Any, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1] / 4):
            s_out += ' '
    return s_out


if __name__ == '__main__':
    # letters = letters_extract(image_file='data/test.png')
    # letters = words_extraction(image_file='data/test.png')

    model = emnist_model()
    fit_model(model)
    model.save('emnist_letters.h5')

    # model = keras.models.load_model('emnist_letters.h5')
    # s_out = img_to_str(model, "hello_world.png")
    # print(s_out)
