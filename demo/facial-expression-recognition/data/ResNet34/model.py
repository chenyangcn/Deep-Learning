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

#coding=utf-8
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
from keras.layers import add,Flatten
#from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
import numpy as np
seed = 7
np.random.seed(seed)

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x



def init_model():
    inpt = Input(shape=(224,224,3))
    x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    #(56,56,64)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    #(28,28,128)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    #(14,14,256)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    #(7,7,512)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dense(1000,activation='softmax')(x)

    model = Model(inputs=inpt,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.summary()
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
