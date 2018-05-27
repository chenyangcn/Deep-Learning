'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import argparse
from model import googlenet_bn # import model file
from data import imagenet
from utils import progress_bar # calculate time using
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--lr', default=0.045, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
net = googlenet_bn() # load model

if device == 'cuda':
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
# print(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if not os.path.isdir('train_log'):
    os.mkdir('train_log')
out_loss = open('./train_log/train_loss_last.txt', 'w')
out_acc = open('./train_log/val_ACC_last.txt', 'w')
# Data
TRAIN_LIST_PATH = './info/train_label.txt'
VALID_LIST_PATH = './info/validation_label.txt'
BATCH_SZIE = 512 # Batch Size

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trainset = imagenet(TRAIN_LIST_PATH, 'train', transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SZIE, shuffle=True, num_workers=16)

testset = imagenet(VALID_LIST_PATH, 'valid', transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SZIE, shuffle=True, num_workers=16)

# learning rate decay
def adjust_learning_rate():
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.94

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        idx = batch_idx + epoch * len(trainloader)
        out_loss.write(str(idx) + ' ' + str(loss.item()) + '\n')

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    out_acc.write(str(epoch) + ' ' + str(acc) + '\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch+1,
        }
        if not os.path.isdir('checkpoint_last'):
            os.mkdir('checkpoint_last')
        torch.save(state, './checkpoint_last/ckpt_v2.t7')
        best_acc = acc

# start_epoch += 1
for epoch in range(start_epoch, 100):
    train(epoch)
    test(epoch)
    if epoch % 2:
        adjust_learning_rate()
