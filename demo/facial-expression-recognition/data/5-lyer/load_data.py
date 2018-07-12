# coding=utf-8
import os
import sys
# import collections
import numpy as np
from PIL import Image
import pandas as pd

training_size = 28709
validation_size = 3589
test_size = 3589
path = './data/fer2013/'


def load_data(data_file='./data/fer2013/fer2013.csv'):
    data = pd.read_csv(data_file)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        #image = Image.fromarray(np.uint8(face))
        #img = image.resize((224, 224), Image.ANTIALIAS)
        #img = img.convert("RGB")
        #face = np.array(img)
        faces.append(face)
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    print('Dataset load success!!')
    # Validation data
    valid_X = faces[training_size : training_size + validation_size]
    valid_Y = emotions[training_size : training_size + validation_size]
    # Test data
    test_X = faces[training_size + validation_size : ]
    test_Y = emotions[training_size + validation_size : ]
    # Training data
    train_X = faces[ : training_size]
    train_Y = emotions[ : training_size]

    print('Dataset init success!!')
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

    #np.save("./data/train_X.npy",train_faces)
    #np.save("./data/train_Y.npy",train_emotions)
    #np.save("./data/valid_X.npy",validation_faces)
    #np.save("./data/valid_Y.npy",validation_emotions)
    #np.save("./data/test_X.npy",test_faces)
    #np.save("./data/test_Y.npy",test_emotions)



if __name__ == "__main__":
    load_data()
