# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:52:15 2020

@author: jayni
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import keras as k
from keras import models, layers
from keras.models import Sequential
import urllib3
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import ImageGrab
import numpy as np
import cv2

urllib3.disable_warnings()
(train_images, train_labels), (test_images, test_labels) = k.datasets.mnist.load_data()

#preposessing the data
#x_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
#x_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
x_train = np.expand_dims(train_images, axis=3)
x_test = np.expand_dims(test_images, axis=3)
# convert class vectors to binary class matrices
y_train = k.utils.to_categorical(train_labels, 10)
y_test = k.utils.to_categorical(test_labels, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#mode 2
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(10, activation='softmax'))

"""
model 1
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
"""

#training model
model.compile(loss=k.losses.categorical_crossentropy,optimizer=k.optimizers.Adadelta(),metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test))

#testing model
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)

"""

train_ds = tfds.load('emnist', split='train')
for example in tfds.as_numpy(train_ds):
  numpy_images, numpy_labels = example["image"], example["label"]

for imgs in numpy_images.split(5):
    plt.imshow(imgs)
"""
