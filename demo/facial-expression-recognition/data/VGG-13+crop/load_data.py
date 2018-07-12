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


def pd_label(label):
    have_to = [0, 1, 2, 4]
    for item in have_to:
        if (item == label):
            return True
    return False


def crop_func(face, work=True):
    crop = [[0, 0], [0, 6], [6, 0], [6, 6], [3, 3]]
    baseLen, endLen = 0, 42
    faces = []
    if work:
        for item in crop:
            temp = face[baseLen+item[0]:endLen +
                        item[0], baseLen+item[1]:endLen+item[1]]
            image = Image.fromarray(np.uint8(temp))
            img = image.resize((224, 224), Image.ANTIALIAS)
            img = img.convert("RGB")
            temp = np.array(img)
            faces.append(temp)
    else:
        image = Image.fromarray(np.uint8(face))
        img = image.resize((224, 224), Image.ANTIALIAS)
        img = img.convert("RGB")
        face = np.array(img)
        faces.append(face)
    return faces


def load_data(data_file='./data/fer2013/fer2013.csv'):
    global training_size
    data = pd.read_csv(data_file)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    emotions = []
    i, k = -1, 0
    for pixel_sequence in pixels:
        i = i+1
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        if (i <= training_size and pd_label(int(data.emotion[i]))):
            face = crop_func(face, True)
            k += 4
            for i in range(5):
                faces.append(face[i])
                emotions.append(int(data.emotion[i]))
        else:
            face = crop_func(face, False)
            faces.append(face)
            emotions.append(int(data.emotion[i]))
    faces = np.asarray(faces)
    emotions = np.asarray(emotions)
    print('Dataset load success!!')
    print(faces.shape)
    print(emotions.shape)
    #faces = np.expand_dims(faces, -1)
    # print(faces[0])
    # print(emotions[0])
    # print(faces.shape)
    training_size += k

    # Validation data
    validation_faces = faces[training_size: training_size + validation_size]
    validation_emotions = emotions[training_size:
                                   training_size + validation_size]
    # Test data
    test_faces = faces[training_size + validation_size:]
    test_emotions = emotions[training_size + validation_size:]
    # Training data
    train_faces = faces[: training_size]
    train_emotions = emotions[: training_size]

    print('Dataset init success!!')

    # return train_faces, train_emotions, validation_faces, validation_emotions, test_faces, test_emotions
    np.save("./data/train_X.npy", train_faces)
    np.save("./data/train_Y.npy", train_emotions)
    np.save("./data/valid_X.npy", validation_faces)
    np.save("./data/valid_Y.npy", validation_emotions)
    np.save("./data/test_X.npy", test_faces)
    np.save("./data/test_Y.npy", test_emotions)


if __name__ == "__main__":
    load_data()
