# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:54:26 2020

@author: jayni
"""


import emnist
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras as k
from keras import models, layers
from keras.models import Sequential
import numpy as np
import pandas as pd

train_images, train_labels = emnist.extract_training_samples("byclass")
test_images, test_labels = emnist.extract_test_samples("byclass")


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
           'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
           'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#pre-processing data
x_train = np.expand_dims(train_images, axis=3)
x_test = np.expand_dims(test_images, axis=3)
# convert class vectors to binary class matrices
y_train = k.utils.to_categorical(train_labels - 1)
y_test = k.utils.to_categorical(test_labels - 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#building the model
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
model.add(Dense(62, activation='softmax'))

#training model
model.compile(loss=k.losses.categorical_crossentropy,optimizer=k.optimizers.Adadelta(),metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

#testing model
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)

#predicting the values
print(x_test.shape)
predval = model.predict(x_test)
plt.imshow(test_images[2000])
print ("Predicted Value:", classes[np.argmax(predval[2000])])
