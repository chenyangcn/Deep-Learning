# coding=utf-8
import os
import sys
import collections
import numpy as np
import pandas as pd
import keras
import h5py
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adadelta
from load_data import load_data
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def init_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(
        224, 224, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))
    # model.load_weights('./data/weights.best.hdf5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model_VGG16.png',show_shapes=True)
    return model


def train_model():
    #train_X, train_Y, valid_X, valid_Y, test_X, test_Y, _ = load_data()
    train_X = np.load('./data/train_X.npy')
    train_Y = np.load('./data/train_Y.npy')
    test_X = np.load('./data/test_X.npy')
    test_Y = np.load('./data/test_Y.npy')
    model = init_model()
    print("OK!", train_X.shape)

    # checkpoint
    filepath = './data/weights1.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    train_X = train_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0
    train_Y = keras.utils.to_categorical(train_Y, 7)
    test_Y = keras.utils.to_categorical(test_Y, 7)
    print(train_Y.shape)

    model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
              nb_epoch=100, batch_size=64, callbacks=callbacks_list, verbose=2)


def find_false_Label():
    test_size = 3589
    index = [0, 0, 0, 0, 0, 0, 0]
    false_label = []
    test_X = np.load('./data/test_X.npy')
    test_Y = np.load('./data/test_Y.npy')
    test_X = test_X.astype('float32') / 255.0
    test_Y = test_Y.astype('float32')
    model = init_model()
    for i in range(test_size):
        loss = model.evaluate(
            test_X[i:i+1], test_Y[i:i+1], batch_size=1, verbose=0)
        #print(' Acc: ', loss[1])
        if (loss[1] < 0.5):
            t = np.dot([0, 1, 2, 3, 4, 5, 6],  test_Y[i].astype('int'))
            false_label.append(i)
            index[t] += 1
            #false_label[t][model.predict(test_X[i]).astype('int')] += 1
            # false_label.append(i)
    data = test_X[false_label, :, :, :]
    pred = model.predict(data)
    print(pred[:5])
    for i in range(7):
        print(i, ' : ', index[i])
    np.save('./data/index.npy', false_label)
    np.save('./data/pred.npy', pred)


if __name__ == "__main__":
    init_model()
    # train_model()
