from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from PIL import Image
import random

VALID_PATH = 'valid/'
TRAIN_PATH = 'train/'

def read_filelist (data_path): #读取filelist
    file_list = open(data_path, 'r')
    imgs = []
    for line in file_list:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append((words[0], int(words[1])))
    return imgs

def load_image (kind, img_name): #加载图片
    if kind == 'train':
        img_path = os.path.join(TRAIN_PATH, img_name)
    else:
        img_path = os.path.join(VALID_PATH, img_name)
    image = Image.open(img_path).convert('RGB')
    return image


class base(Dataset):
    def __init__(self, data_path, kind, transform=None, loader=load_image):
        self.transform = transform
        self.kind = kind
        self.imgs = read_filelist(data_path)
        self.loader = loader

    def __getitem__(self, index): # 获取一个index
        img_name, label = self.imgs[index]
        image = self.loader(self.kind, img_name)
        image = self.transform(image)
        return image, label

    def __len__(self): # 输出每一个epoch的迭代数
        return len(self.imgs)


class  imagenet(base): # 继承前面实现的base数据集
    def __getitem__(self, index):
        try:
            # 调用父类的获取函数
            return super(imagenet, self).__getitem__(index)
        except:
            new_index = random.randint(0, len(self) - 1)
            return self[new_index]
