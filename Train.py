# Deepfool Attack on CIFAR10

import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torch.nn import DataParallel

from Denoising_ResNet import Denoising_ResNet18

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='ResNet18', help="model: WideResNet / ResNet18,34,50,101,152 / DenseNet121,169,201,161 / DPN26,92 / EfficientNetB0 / GoogLeNet / LeNet / MobileNet / MobileNetV2 / PNASNetA,B / PreActResNet18,34,50,101,152 / ResNeXt29_2x64d / ResNeXt29_4x64d / ResNeXt29_8x64d / ResNeXt29_32x4d / SENet18 / ShuffleNetG2,G3 / ShuffleNetV2 / VGG11,13,16,19")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning_rate")
parser.add_argument('--epochs', type=int, default=200, help="epoch")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--num_workers', type=int, default=32, help="num_workers")

args = parser.parse_args()

def test(model, testloader, criterion):
    model.eval()
    correct, total, loss, counter = 0, 0, 0, 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1
    return loss / counter, correct / total

# Set the transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the dataset
trainset = torchvision.datasets.CIFAR10(root='/home/eva_share/datasets/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=32)
testset = torchvision.datasets.CIFAR10(root='/home/eva_share/datasets/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = Denoising_ResNet18()
model = model.cuda()
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
params = list(model.parameters())
optimizer = optim.SGD(params, lr=args.learning_rate, weight_decay=1e-4, momentum=0.9, nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(args.epochs*3/4), int(args.epochs*7/8)], gamma=0.1)

with open(args.model + 'training_log.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "iteration", "train_loss", "val_loss", "acc", "val_acc"])

i, total, correct, train_loss, counter = 0, 0, 0, 0, 0

for epoch in range(args.epochs):
    for data in trainloader:
        model.train()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # count acc,loss on trainset
        _, predicted = torch.max(outputs.data, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()
        train_loss += loss.item()
        counter += 1
        i += 1
        if i % 400 == 0:
            acc = correct / total
            train_loss /= counter
            val_loss, val_acc = test(model, testloader, criterion)
            with open(args.model + 'training_log.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, i, train_loss, val_loss, acc, val_acc])
            print("epoch:{}, i:{}, train_loss:{:.3f}, train_acc:{:.3f}, val_loss:{:.3f}, val_acc:{:.3f}".format(epoch, i, train_loss, acc, val_loss, val_acc))
            correct, total, train_loss, counter = 0, 0, 0, 0
        torch.save(model.state_dict(), args.model + '.pkl')
