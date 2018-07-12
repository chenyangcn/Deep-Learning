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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def init_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(1, 1), input_shape=(
        48, 48, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(384, activation='relu'))
    model.add(Dense(192, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # model.load_weights('./data/weights.best.hdf5')
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model_VGG16.png',show_shapes=True)
    return model


def train_model():
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = load_data()
    model = init_model()
    #train_X = np.load('./data/tain_X.npy')
    #train_Y = np.load('./data/train_Y.npy')
    #test_X = np.load('./data/test_X.npy')
    #test_Y = np.load('./data/test_Y.npy')
    print("OK!")

    # checkpoint
    filepath = './data/weights1.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    train_X = train_X.astype('float32') / 255.0
    valid_X = valid_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0
    train_Y = train_Y.astype('float32')
    valid_Y = valid_Y.astype('float32')
    test_Y = test_Y.astype('float32')

    model.fit(train_X, train_Y, validation_data=(valid_X, valid_Y),
              nb_epoch=100, batch_size=128, callbacks=callbacks_list, verbose=2)


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
    # train_model()
    init_model()
